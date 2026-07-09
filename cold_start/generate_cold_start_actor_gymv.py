#!/usr/bin/env python
"""
Cold-start actor-agent rollouts for **gym-v Temporal** games (gpt-5.5 vision pipeline).

Pipeline (one outer step):

  1. Reset/step a ``gym_v.make("Temporal/<Title>-v0")`` env, pulling the
     multimodal :class:`Observation` (``obs.image`` PIL frame, ``obs.text``,
     ``obs.metadata``).
  2. Visual grounding: gpt-5.5 (vision) converts the frame into a canonical
     ``<state>...</state>`` schema using ``vlm_wrapper.schema``.  The wrapper's
     auxiliary text observation rides along as supporting context only.
  3. Action selection: gpt-5.5 reads the schema + the env's
     ``obs.metadata["available_actions"]`` list and picks ONE action via
     OpenAI function calling.
  4. ``env.step({agent_id: action})`` and an :class:`Experience` is appended
     to the :class:`Episode`.  The schema, raw VLM output, action reasoning,
     and reward are all preserved on the Experience so SFT/GRPO consumers
     can replay the trajectory exactly.

Companion to ``cold_start/generate_cold_start_actor.py`` (env_wrappers /
GamingAgent / Orak) — same Episode/Experience output format, but driven
through the gym-v multi-agent observation API and stable-retro / Genesis
ROMs registered as ``Temporal/<Title>-v0``.

Output layout (``<codebase_root>/Cold-start-out-gymv/<env_id_safe>/``):

  - ``episode_NNN.json``       individual Episode (Episode.to_dict())
  - ``episode_buffer.json``    Episode_Buffer (loadable for trainer)
  - ``rollouts.jsonl``         append-only JSONL, one Episode per line
  - ``rollout_summary.json``   per-env stats
  - ``frames/<ep>/step_NNN.png``  rendered frames sent to the VLM (debug)

Usage (from the Multi-hop-Reasoning-VLM-Agent root)::

    export OPENAI_API_KEY="sk-..."          # or OPENROUTER_API_KEY

    # Default: 1 episode of Airstriker, 30 steps
    python cold_start/generate_cold_start_actor_gymv.py

    # Two Temporal envs, 3 episodes each, 60 steps
    python cold_start/generate_cold_start_actor_gymv.py \\
        --envs Temporal/Airstriker-v0 Temporal/SpaceHarrierII-v0 \\
        --episodes 3 --max_steps 60 -v

    # Skip the vision call (cheap canonical-schema baseline)
    python cold_start/generate_cold_start_actor_gymv.py --no_vision

    # Resume an interrupted run
    python cold_start/generate_cold_start_actor_gymv.py --resume
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
# Path setup — make the codebase, sibling repos, and gym-v importable.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent
GYMV_ROOT = WORKSPACE_ROOT / "gym-v"

for _p in [str(CODEBASE_ROOT), str(GYMV_ROOT)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


def _bootstrap_api_keys_from_file() -> Optional[Path]:
    """Seed ``os.environ`` from a sibling ``api_keys.py`` if present.

    Looked-up locations (first hit wins):
      1. ``$COSPLAY_API_KEYS_FILE``
      2. ``cold_start/api_keys.py``
      3. ``<codebase_root>/api_keys.py``
      4. ``<codebase_root>/../api_keys.py``  (workspace root)
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
except Exception:  # pragma: no cover — keep script runnable in isolation
    _SFT_TEACHER_MODEL = "gpt-5.5"

DEFAULT_MODEL = _SFT_TEACHER_MODEL  # gpt-5.5

try:
    from API_func import OPENROUTER_BASE, make_openai_client, effective_openai_model
except Exception:  # pragma: no cover
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    make_openai_client = None
    effective_openai_model = None

logger = logging.getLogger("cold_start.actor_gymv")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fallback default env ids — these are stable-retro / Genesis ROMs registered
# in gym-v via gym_v.envs.multi_turn.temporal.retro_env. Whether they
# actually run depends on whether you've imported the ROM into stable-retro.
DEFAULT_ENVS: List[str] = [
    "Temporal/Airstriker-v0",
]

# Per-env step caps. Genesis frames are cheap to step; we cap at 60 by default
# so a vision-call-per-step run terminates in reasonable time.
DEFAULT_MAX_STEPS = 60
# Default episode count per env when ``--episodes`` is not given.
DEFAULT_EPISODES = 1
# Per-agent-step emulator frame budget. The Genesis runs at 60 fps; with
# ``frame_skip=1`` (the original behaviour) every agent decision advances
# the emulator by exactly one frame, which means a 100-step episode is
# only 1.67 s of real game time — far too short for retro games whose
# title-screen-to-first-reward window is 5–10 s. Setting ``frame_skip=8``
# matches the standard Atari/Retro RL convention (Mnih+2015 used 4 for
# Atari; Genesis games run faster and need 8). The agent's action mask
# is held for ``frame_skip`` consecutive emulator frames and the per-frame
# reward is summed before the next agent decision.
DEFAULT_FRAME_SKIP = 1

# Anti-noop: force a different action after this many consecutive steps
# whose state is identical AND reward ≤ 0.
_MAX_CONSECUTIVE_NOOPS = 3
# Number of recent action results to surface in the action-selection prompt.
_HISTORY_WINDOW = 5
# Default token budgets.
# Strict-enum tool_call output is ~7–14 tokens with `{"action": "<name>"}`
# (see `_build_action_tools`).  128 gives ~10× safety headroom while
# leaving plenty of input context: 9B vLLM is served at
# --max-model-len 8192 and dense gym-v schemas can push the prompt
# above 7.5 K tokens, so a smaller output budget here avoids
# `BadRequestError: This model's maximum context length is 8192`.
# Reasoning models get `max(6000, max_tokens*4)` in `_chat_completion`,
# so they remain unaffected.
_ACTION_MAX_TOKENS = 128
_SCHEMA_MAX_TOKENS = 4000
# Reasoning models burn output tokens on hidden thinking — give them more.
_SCHEMA_MAX_TOKENS_REASONING = 12000

# Models that require ``max_completion_tokens`` (no ``temperature``).
# Matches gpt-5.x, gpt-5.5*, o1/o3/o4 families.
_REASONING_MODEL_RE = re.compile(
    r"(?:^|/)(?:gpt-5(?:[\.\-]\w+)?|o[134](?:[\.\-]\w+)?)(?:$|[^\w])",
    re.IGNORECASE,
)


def _is_reasoning_model(model: str) -> bool:
    """Return True for OpenAI-style reasoning models (gpt-5.x, o1/o3/o4)."""
    if not model:
        return False
    return bool(_REASONING_MODEL_RE.search(model))


# Models that secretly burn tokens on hidden chain-of-thought *before* the
# tool_call payload but still accept the standard ``max_tokens`` /
# ``temperature`` knobs.  Matches Anthropic Claude 4.x+ (extended thinking
# enabled by default on Sonnet 4.5+, Opus 4.x), Google Gemini 2.5/3.x
# (always-on hidden reasoning on Pro tier), and Qwen3 / Qwen3.5 / Qwen3.6
# (ditto — see below).  Without bumping the output budget here, an action
# call with the strict-enum tool_choice will return ``finish_reason=length``
# and an empty ``tool_calls`` once the hidden reasoning trace fills 128
# tokens, silently degrading to the random-action fallback (observed on
# google/gemini-3.1-pro-preview AND on qwen/qwen3.5-9b +
# qwen/qwen3.5-35b-a3b via OpenRouter on 2026-05-03 — the
# ``extra_body.enable_thinking=False`` knob is silently ignored on the
# OpenRouter Qwen3.5 routes, so the only reliable mitigation is a wide
# output budget).
_THINKING_MODEL_RE = re.compile(
    r"(?:^|/)("
    r"claude-(?:sonnet-|opus-|haiku-)?(?:[4-9]|\d{2,})"   # claude-4.x+, claude-sonnet-4.x+
    r"|gemini-(?:[2-9]|\d{2,})"                           # gemini-2.x+, gemini-3.x+
    r"|qwen3(?:\.\d+)?(?=[-./]|$)"                        # qwen3-, qwen3.5-, qwen3.6-, qwen3-max, qwen3-vl-*, …
    r")",
    re.IGNORECASE,
)


def _is_thinking_model(model: str) -> bool:
    """Return True for hosted thinking-class models that need a wide
    output budget but otherwise behave like classic OpenAI-style chat APIs.
    """
    if not model:
        return False
    return bool(_THINKING_MODEL_RE.search(model))


def _sanitize_env_id(env_id: str) -> str:
    """Filesystem-safe rendering of a Gym-V env id."""
    return re.sub(r"[^\w\-.]+", "_", env_id)


# ---------------------------------------------------------------------------
# Image / observation helpers
# ---------------------------------------------------------------------------

def _to_pil(image: Any):
    """Coerce ``obs.image`` (PIL.Image | numpy | list[PIL]) into a single PIL RGB."""
    try:
        from PIL import Image
    except ImportError:
        return None
    if image is None:
        return None
    if isinstance(image, list) and image:
        image = image[-1]
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
    """Cross-model chat-completion wrapper.

    Reasoning models (gpt-5.x, o1/o3/o4) reject ``max_tokens`` and
    ``temperature``; they require ``max_completion_tokens`` and burn part
    of that budget on hidden thinking tokens. We detect reasoning models
    up front and route them through with a generous output cap. Classic
    models keep the legacy path with a single fallback retry.

    ``reasoning_effort`` (one of ``minimal`` / ``low`` / ``medium`` / ``high``)
    is forwarded only for reasoning models; ignored otherwise.  Setting
    ``minimal`` suppresses hidden thinking tokens — the right default for
    cold-start data generation, where the SFT student only learns from
    the visible ``<state>`` and action JSON.
    """
    if _is_reasoning_model(model):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max(6000, max_tokens * 4),
        }
        if reasoning_effort:
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

    # Thinking-class hosted models (Claude 4.x+, Gemini 2.x+/3.x+) emit a
    # hidden chain-of-thought *before* the tool_call.  Bump max_tokens so
    # that hidden trace cannot truncate the tool_call (otherwise we get
    # finish_reason=length, tool_calls=None, and silent random fallback).
    # Temperature is preserved — these models accept it normally.
    effective_max_tokens = max_tokens
    if _is_thinking_model(model):
        effective_max_tokens = max(6000, max_tokens * 4)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    # Thinking-mode-class models (Qwen3*, Qwen3.5*, DeepSeek-R1-distill, …)
    # emit a free-form `<think>...</think>` block before the tool call AND,
    # when served by Alibaba DashScope (or proxied through OpenRouter),
    # reject strict ``tool_choice={"type":"function",...}`` payloads with
    # HTTP 400 ("InvalidParameter ... in thinking mode").  Disable thinking
    # via *both* recognised parameter names so the same payload works
    # regardless of routing — the server silently ignores the unknown key:
    #
    #   - DashScope / OpenRouter:   ``extra_body.enable_thinking = False``
    #   - vLLM-OpenAI-compat:       ``extra_body.chat_template_kwargs
    #                                .enable_thinking = False``
    #
    # Heuristic: model id contains a slash (HuggingFace ``<org>/<name>``
    # for vLLM **or** OpenRouter ``<provider>/<slug>``).  Managed APIs
    # like ``gpt-4o`` / ``claude-3.5-sonnet`` / ``o3-mini`` do not.
    if "/" in model:
        kwargs["extra_body"] = {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

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
        kwargs.pop("extra_body", None)
        kwargs["max_completion_tokens"] = max(6000, max_tokens * 5)
        return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Lazy imports for optional deps
# ---------------------------------------------------------------------------

def _import_gymv_stack():
    """Import gym_v + per-game visual-grounding helpers."""
    import gym_v
    from gymv_wrapper.temporal_visual_grounding import (
        TEMPORAL_GAME_SPECS,
        build_temporal_visual_schema,
    )
    return gym_v, TEMPORAL_GAME_SPECS, build_temporal_visual_schema


def _import_schema_helpers():
    """Lazy import of ``vlm_wrapper.schema`` helpers."""
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


def _rom_resolves(retro_game: str) -> bool:
    """Return True iff stable-retro can locate the ROM file for *retro_game*."""
    try:
        import stable_retro  # type: ignore
    except Exception:
        return False
    try:
        stable_retro.data.get_romfile_path(retro_game)
        return True
    except (FileNotFoundError, OSError):
        if retro_game.endswith("-v0"):
            return False
        try:
            stable_retro.data.get_romfile_path(f"{retro_game}-v0")
            return True
        except (FileNotFoundError, OSError):
            return False


# ---------------------------------------------------------------------------
# Heuristic → <state> string fallback
# ---------------------------------------------------------------------------

def _heuristic_to_state_block(
    grounding: Dict[str, Any],
    *,
    domain: str,
    task: str,
    goal: str,
    step: int,
    actions: List[str],
) -> str:
    """Render the dict from ``build_temporal_visual_schema`` as a ``<state>`` block.

    This is a best-effort serializer used as the deterministic fallback when
    the VLM call fails. It only covers the bits of the heuristic schema that
    the actor needs (entities + a few HUD signals) — anything richer is
    persisted separately on the Experience.
    """
    lines: List[str] = []
    lines.append("<state>")
    lines.append(f"domain={domain}")
    lines.append(f"task={task}")
    lines.append(f"goal={goal}")
    lines.append(f"step={step}")
    lines.append("")

    entities = list(grounding.get("entities") or [])
    lines.append("<entities>")
    eid_map: Dict[str, str] = {}
    for i, ent in enumerate(entities, start=1):
        new_id = f"e{i}"
        eid_map[ent.get("eid", new_id)] = new_id
        ent_type = ent.get("type", "object")
        label = ent.get("label", "unknown")
        bid = ent.get("bid")
        bid_str = "null" if bid is None else str(bid)
        pos = ent.get("pos")
        pos_str = "null" if pos is None else str(pos)
        ontology = ent.get("ontology", "tracked_entity")
        lines.append(
            f"{new_id}[type={ent_type}, label={label}, "
            f"bid={bid_str}, pos={pos_str}, ontology={ontology}]"
        )
    lines.append("")

    lines.append("<attributes>")
    for ent in entities:
        new_id = eid_map.get(ent.get("eid", ""), "")
        if not new_id:
            continue
        if "value" in ent and ent["value"] is not None:
            lines.append(f"{new_id}.value={ent['value']}")
        lines.append(f"{new_id}.state=visible")
    lines.append("")

    lines.append("<affordances>")
    for ent in entities:
        new_id = eid_map.get(ent.get("eid", ""), "")
        if not new_id:
            continue
        ontology = ent.get("ontology", "")
        if ontology == "navigable_region":
            verbs = "navigate_to, inspect"
        elif ontology == "goal_indicator":
            verbs = "read, track"
        else:
            verbs = "track, inspect"
        lines.append(f"{new_id}.affords=[{verbs}]")
    lines.append("")

    lines.append("<relations>")
    lines.append("")

    sim = grounding.get("simulation") or {}
    ep_reward = sim.get("episode_reward")
    if isinstance(ep_reward, (int, float)) and ep_reward > 0:
        phase = "mid"
    else:
        phase = "early"
    lines.append("<state_flags>")
    lines.append("progress=null")
    lines.append(f"phase={phase}")
    lines.append("scene_type=game_play")
    lines.append("error=null")
    lines.append("dialog_open=false")
    lines.append("input_pending=true")
    lines.append("")

    lines.append("<targets>")
    lines.append("target=null")
    lines.append("blocker=null")
    lines.append("constraint=null")
    lines.append("candidate_set=[]")
    lines.append("history_anchor=null")
    lines.append("")

    lines.append("<actions>")
    for i, a in enumerate(actions or [], start=1):
        lines.append(f"a{i}={a}")
    lines.append("</state>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1 — visual schema generation (gpt-5.5 vision)
# ---------------------------------------------------------------------------

# Pre-compiled patterns for the lenient parser.
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


def generate_schema_from_image(
    *,
    pil_image,
    obs_text: str,
    env_id: str,
    display_name: str,
    grounding_focus: str,
    task_id: str,
    goal: str,
    step: int,
    valid_actions: List[str],
    client: Any,
    routed_model: str,
    schema_helpers: Dict[str, Any],
    canonical_fallback: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = _SCHEMA_MAX_TOKENS,
    max_entities: int = 25,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call gpt-5.5 (vision) to produce a ``<state>...</state>`` schema.

    The image is the primary input; ``obs_text`` rides along as auxiliary
    context. Returns a dict with the parsed ``schema`` (or ``None``), the
    raw model output, the routed model id, and any exception captured.

    When the API call fails or no schema is parsed, falls back to the
    deterministic ``canonical_fallback`` if provided.
    """
    if pil_image is None or schema_helpers is None or client is None:
        return {
            "schema": canonical_fallback,
            "raw": "",
            "source": "fallback_canonical" if canonical_fallback else "no_image_or_client",
            "error": None,
        }

    system = schema_helpers["build_system_prompt"]("gymv", max_entities=max_entities)

    extra_parts: List[str] = [
        f"Game info: {display_name} ({env_id}). "
        f"Grounding focus: {grounding_focus or 'n/a'}."
    ]
    if obs_text:
        extra_parts.append(
            "Environment text observation (auxiliary — for reference only):\n"
            f"{obs_text[:4000]}"
        )
    if valid_actions:
        extra_parts.append(
            "Valid actions for this environment (you MUST copy these strings "
            "verbatim into <actions>; do NOT rename or reformat):\n"
            + "\n".join(f"  - {a}" for a in valid_actions[:25])
        )
    extra_context = "\n\n".join(extra_parts)

    user_content = schema_helpers["build_user_message"](
        pil_image,
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
        logger.warning("[schema-VLM] %s step %d failed: %s", env_id, step, exc)

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
                "[schema-VLM] %s step %d salvaged via '%s' (finish_reason=%s, raw_len=%d)",
                env_id, step, recovery, finish_reason, len(raw),
            )
        return {"schema": parsed, "source": "vlm", **base_meta}

    if raw and finish_reason == "length":
        logger.warning(
            "[schema-VLM] %s step %d response truncated (finish_reason=length, raw_len=%d) — "
            "consider raising _SCHEMA_MAX_TOKENS",
            env_id, step, len(raw),
        )

    return {
        "schema": canonical_fallback,
        "source": "fallback_canonical" if canonical_fallback else "vlm_no_schema",
        **base_meta,
    }


# ---------------------------------------------------------------------------
# Stage 2 — schema-driven action selection (gpt-5.5)
# ---------------------------------------------------------------------------

_ACTOR_SYSTEM_PROMPT = (
    "You are an Actor Agent for the COS-PLAY game-AI pipeline, driving a "
    "gym-v Temporal (stable-retro / Genesis) environment.\n"
    "On every step you receive a structured ``<state>...</state>`` schema "
    "describing the current visual state of the game, plus the set of "
    "valid actions the environment will accept this step.\n\n"
    "Your job:\n"
    "1. Reason briefly (≤3 sentences) about the schema: which entities "
    "matter, what is the current sub-goal, and why one action best "
    "advances the task.\n"
    "2. Pick EXACTLY ONE action from the valid-action list — copy the "
    "string verbatim (no renaming, no quoting, no reformatting).\n\n"
    "If recent action history shows an action had NO EFFECT (state did "
    "not change and reward ≤ 0), choose a DIFFERENT action this turn.\n\n"
    "Always respond by calling the ``choose_action`` function."
)


def _build_action_tools(action_names: List[str]) -> list:
    """OpenAI function-calling tool definition for action selection.

    Minimal strict-enum schema — only the ``action`` field with the full
    ``enum`` of valid action names — so the model cannot generate
    free-form text that overruns the token budget.

    The earlier optional ``reasoning`` (chain-of-thought) field was
    dropped because vLLM-served thinking models (Qwen3 / Qwen3.5,
    DeepSeek-R1-distill, …) wrote multi-paragraph reasoning into it,
    blew past ``_ACTION_MAX_TOKENS``, and truncated the JSON
    mid-string ("Unterminated string starting at: line N column M").
    Reasoning was only used for log diagnostics, not metrics or
    training data, so removing it is the cleaner fix.

    Backends that do constrained / guided generation (vLLM with
    ``--enable-auto-tool-choice --tool-call-parser hermes``) honor the
    ``enum`` and refuse to emit invalid tokens.  Other backends still
    get the description as a hint and the actor's downstream
    ``_canonicalize_action`` accepts reasonable spellings.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "description": "Choose the single environment action for this turn.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": action_names,
                            "description": (
                                "EXACT verbatim string from the valid actions list. "
                                f"Allowed: {', '.join(action_names[:25])}"
                            ),
                        },
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _format_history_block(history: List[Dict[str, Any]]) -> str:
    if not history:
        return ""
    lines = ["Recent action history (newest last):"]
    for entry in history[-_HISTORY_WINDOW:]:
        effect = "NO EFFECT" if entry.get("noop") else f"reward {entry.get('reward', 0.0):+.2f}"
        lines.append(f"  - {entry.get('action')!r} -> {effect}")
    noop_actions = sorted({e["action"] for e in history[-_HISTORY_WINDOW:] if e.get("noop")})
    if noop_actions:
        lines.append(f"WARNING: Action(s) {noop_actions} had no effect. Pick a DIFFERENT action.")
    return "\n".join(lines) + "\n"


def _canonicalize_action(raw: str, action_names: List[str]) -> Optional[str]:
    """Map ``raw`` to one of ``action_names`` (case-insensitive, prefix, substring)."""
    if not raw:
        return None
    cand = raw.strip().strip("`").strip('"').strip("'")
    if not cand:
        return None

    if cand in action_names:
        return cand
    lc = cand.lower()
    lower_map = {a.lower(): a for a in action_names}
    if lc in lower_map:
        return lower_map[lc]

    m = re.match(r"^\s*(\d+)\s*[\.\)\-:]?\s*$", cand)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(action_names):
            return action_names[idx]

    for a in action_names:
        if a.lower() == lc:
            return a
    for a in action_names:
        if a.lower().startswith(lc) or lc.startswith(a.lower()):
            return a
    for a in action_names:
        if a.lower() in lc or lc in a.lower():
            return a
    return None


def select_action_from_schema(
    *,
    schema_text: Optional[str],
    obs_text: str,
    valid_actions: List[str],
    task: str,
    env_id: str,
    step: int,
    history: List[Dict[str, Any]],
    client: Any,
    routed_model: str,
    temperature: float = 0.4,
    max_tokens: int = _ACTION_MAX_TOKENS,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Call gpt-5.5 with the schema → ``(action, reasoning, raw, error)``."""
    if not valid_actions:
        return None, None, "", "no_valid_actions"
    if client is None:
        return None, None, "", "no_client"

    history_block = _format_history_block(history)
    schema_block = (
        schema_text.strip() if schema_text else
        "(no schema available — fall back to the auxiliary text observation)"
    )

    user_parts = [
        f"Task: {task}",
        f"Env: {env_id}",
        f"Step: {step}",
        "",
        "Structured state schema:",
        schema_block,
    ]
    if not schema_text and obs_text:
        user_parts.extend([
            "",
            "Auxiliary text observation (since no schema is available):",
            obs_text[:3000],
        ])
    user_parts.extend([
        "",
        f"Valid actions: {', '.join(valid_actions)}",
        "",
        history_block.strip(),
        "",
        "Think step-by-step about the schema entities and pick the BEST action. "
        "Then call the choose_action function.",
    ])
    user_content = "\n".join(p for p in user_parts if p is not None)

    tools = _build_action_tools(valid_actions)

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
            action_raw = str(args.get("action", "")).strip()
            reasoning = args.get("reasoning") or None
            canonical = _canonicalize_action(action_raw, valid_actions)
            if canonical:
                return canonical, reasoning, raw or json.dumps(args), None

        canonical = _canonicalize_action(raw, valid_actions)
        if canonical:
            return canonical, None, raw, None

    except Exception as exc:
        err = repr(exc)
        logger.warning("[action-LLM] %s step %d failed: %s", env_id, step, exc)

    return None, None, raw, err


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _is_noop(prev_text: str, next_text: str, reward: float) -> bool:
    """Best-effort no-op detection: state unchanged AND non-positive reward."""
    if reward and float(reward) > 0.0:
        return False
    return (prev_text or "") == (next_text or "")


def _pick_different(action: str, candidates: List[str]) -> str:
    alts = [a for a in candidates if a != action]
    return random.choice(alts) if alts else action


def _resolve_goal_line(env, env_id: str) -> str:
    """Pick a one-line goal from env.description, falling back to the env id."""
    description = ""
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "description"):
        try:
            description = getattr(unwrapped, "description", "") or ""
        except Exception:
            description = ""
    if isinstance(description, dict):
        # Multi-agent envs sometimes return a per-agent dict.
        first = next(iter(description.values()), "")
        description = first if isinstance(first, str) else str(first)
    if isinstance(description, str) and description.strip():
        return description.strip().split("\n")[0]
    if getattr(env, "spec", None) is not None:
        return str(env.spec.id or env_id)
    return env_id


def run_actor_episode(
    *,
    env_id: str,
    spec: Any,
    max_steps: int,
    client: Any,
    routed_model: str,
    fallback_model: str,
    schema_helpers: Optional[Dict[str, Any]],
    use_vision: bool,
    temperature_action: float,
    temperature_schema: float,
    frames_dir: Optional[Path],
    seed: Optional[int],
    verbose: bool,
    step_stream_path: Optional[Path] = None,
    ep_idx: int = 0,
    reasoning_effort: Optional[str] = None,
    frame_skip: int = DEFAULT_FRAME_SKIP,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode end-to-end and return ``(Episode, stats)``.

    If ``step_stream_path`` is provided, every completed step is flushed to
    that path as a single JSON line *immediately* after the env step. This
    makes the rollout crash-safe: a SIGTERM / OOM / API outage mid-episode
    will preserve every step that finished before the failure. The stream is
    truncated at episode start so a re-run via ``--resume`` cannot mix old
    partial data into a fresh attempt.
    """
    gym_v, _, build_temporal_visual_schema = _import_gymv_stack()

    # Truncate any stale partial stream from a previous crashed attempt so
    # we never blend old half-data with the fresh re-run.
    if step_stream_path is not None:
        step_stream_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            step_stream_path.write_text("", encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Could not truncate step stream %s: %s", step_stream_path, exc
            )

    env = gym_v.make(env_id)
    if frame_skip and frame_skip > 1:
        # Hold every agent action for ``frame_skip`` emulator frames and
        # sum the per-frame reward. ``stickprob=0.0`` makes the wrapper
        # deterministic (no Atari-style sticky actions); we want the
        # action selected by the LLM to apply uniformly across the skip
        # window so the recorded `experiences` log stays coherent.
        try:
            from gym_v.wrappers import StochasticFrameSkip
            env = StochasticFrameSkip(env, n=frame_skip, stickprob=0.0)
        except Exception as exc:                                    # noqa: BLE001
            logger.warning(
                "frame_skip=%d requested but StochasticFrameSkip unavailable (%s); "
                "falling back to skip=1.",
                frame_skip, exc,
            )
    try:
        odict, info_dict = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        # Older Gymnasium-compat envs that don't accept seed=
        odict, info_dict = env.reset()
    agent_id = next(iter(odict))
    obs = odict[agent_id]

    display_name = getattr(spec, "display_name", env_id)
    grounding_focus = getattr(spec, "grounding_focus", "")
    genre = getattr(spec, "genre", "unknown")
    task_id = env_id
    goal = _resolve_goal_line(env, env_id)
    task = (
        f"Play {display_name} ({genre}). Visual focus: {grounding_focus or 'n/a'}. "
        f"Goal: {goal}"
    )

    experiences: List[Experience] = []
    history: List[Dict[str, Any]] = []
    consecutive_noops = 0
    last_noop_action: Optional[str] = None
    schema_calls = 0
    schema_ok = 0
    action_llm_ok = 0
    action_llm_fail = 0
    total_reward = 0.0
    terminated_all = False
    truncated_all = False

    t0 = time.time()
    for step in range(max_steps):
        meta = dict(getattr(obs, "metadata", None) or {})
        valid_actions: List[str] = [
            str(a) for a in (meta.get("available_actions") or [])
        ][:25]
        if not valid_actions:
            # No action vocab — emit NOOP and continue (Genesis envs always
            # accept "NOOP" but we still prefer to expose what's available).
            valid_actions = ["NOOP"]

        # 1. Pull a frame (PIL) for the VLM.
        pil = _to_pil(getattr(obs, "image", None))
        img_path: Optional[str] = None
        if pil is not None and frames_dir is not None:
            img_path = _save_frame(pil, frames_dir / f"step_{step:03d}.png")

        # 2. Heuristic visual grounding (deterministic, modality-fused).
        try:
            heuristic = build_temporal_visual_schema(env_id, obs)
        except Exception as exc:
            logger.debug("build_temporal_visual_schema(%s) failed: %s", env_id, exc)
            heuristic = {}

        canonical_schema = _heuristic_to_state_block(
            heuristic,
            domain="gymv",
            task=task_id,
            goal=goal,
            step=step,
            actions=valid_actions,
        )

        # 3. Visual grounding (vision call): frame → schema.
        schema_text: Optional[str] = None
        schema_meta: Dict[str, Any] = {
            "schema": None, "raw": "", "source": "skipped", "error": None,
        }
        if use_vision and pil is not None and schema_helpers is not None and client is not None:
            schema_budget = (
                _SCHEMA_MAX_TOKENS_REASONING
                if _is_reasoning_model(routed_model)
                else _SCHEMA_MAX_TOKENS
            )
            schema_meta = generate_schema_from_image(
                pil_image=pil,
                obs_text=getattr(obs, "text", None) or "",
                env_id=env_id,
                display_name=display_name,
                grounding_focus=grounding_focus,
                task_id=task_id,
                goal=goal,
                step=step,
                valid_actions=valid_actions,
                client=client,
                routed_model=routed_model,
                schema_helpers=schema_helpers,
                canonical_fallback=canonical_schema,
                temperature=temperature_schema,
                max_tokens=schema_budget,
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
                "source": "canonical",
                "error": None,
            }

        # 4. Action selection (text-only call: schema → action).
        obs_text_str = getattr(obs, "text", None) or ""
        action, reasoning, action_raw, action_err = select_action_from_schema(
            schema_text=schema_text,
            obs_text=obs_text_str,
            valid_actions=valid_actions,
            task=task,
            env_id=env_id,
            step=step,
            history=history,
            client=client,
            routed_model=routed_model,
            temperature=temperature_action,
            reasoning_effort=reasoning_effort,
        )
        if action is not None:
            action_llm_ok += 1
        else:
            action_llm_fail += 1
            action = valid_actions[0]
            if last_noop_action == action and len(valid_actions) > 1:
                action = _pick_different(action, valid_actions)

        # 5. Anti-NOOP override.
        if (
            consecutive_noops >= _MAX_CONSECUTIVE_NOOPS
            and action == last_noop_action
            and len(valid_actions) > 1
        ):
            old_action = action
            action = _pick_different(action, valid_actions)
            reasoning = (
                (reasoning or "") + f" [auto-override: '{old_action}' was no-op {consecutive_noops}x]"
            )
            if verbose:
                print(f"  step {step}: anti-noop override {old_action!r} -> {action!r}")

        # 6. Step the env (gym-v multi-agent dict API).
        try:
            odict, reward_dict, term_dict, trunc_dict, next_info_dict = env.step(
                {agent_id: action}
            )
        except Exception as exc:
            logger.error("[%s] step %d env.step(%r) failed: %s", env_id, step, action, exc)
            if verbose:
                traceback.print_exc()
            break

        if isinstance(reward_dict, dict):
            r = reward_dict.get(agent_id, reward_dict.get("__all__", 0.0))
        else:
            r = reward_dict
        try:
            reward = float(r or 0.0)
        except (TypeError, ValueError):
            reward = 0.0
        total_reward += reward

        terminated_all = (
            bool(term_dict.get("__all__", False)) if isinstance(term_dict, dict)
            else bool(term_dict)
        )
        truncated_all = (
            bool(trunc_dict.get("__all__", False)) if isinstance(trunc_dict, dict)
            else bool(trunc_dict)
        )
        done = terminated_all or truncated_all

        # Observation rotation (auto-switch — pick the next agent if the env
        # cycles through agents; for single-agent Temporal envs this is a no-op).
        if odict:
            agent_id = next(iter(odict))
            next_obs = odict[agent_id]
        else:
            next_obs = obs
        next_obs_text = getattr(next_obs, "text", None) or ""

        is_noop = _is_noop(obs_text_str, next_obs_text, reward)
        history.append({"action": action, "reward": reward, "noop": is_noop})

        if is_noop and action == last_noop_action:
            consecutive_noops += 1
        elif is_noop:
            consecutive_noops = 1
            last_noop_action = action
        else:
            consecutive_noops = 0
            last_noop_action = None

        # 7. Build the Experience record.
        exp = Experience(
            state=obs_text_str,
            action=str(action),
            reward=reward,
            next_state=next_obs_text,
            done=done,
            intentions=reasoning,
            tasks=task,
        )
        exp.idx = step
        exp.action_type = "primitive"
        # Persist truncated raw observation snapshots for replay.
        exp.raw_state = obs_text_str[:4000] if obs_text_str else None
        exp.raw_next_state = next_obs_text[:4000] if next_obs_text else None
        exp.available_actions = list(valid_actions)
        exp.interface = {
            "env_name": "gym_v",
            "game_name": display_name,
            "env_id": env_id,
            "wrapper": "TemporalVisualGroundingWrapper",
        }
        # Stash schema + VLM outputs both on Experience.extras (in-memory) AND
        # Experience.metadata (Experience.to_dict serialises only metadata, so
        # mirroring keeps the trajectory replayable from disk).
        extras: Dict[str, Any] = {
            "schema": schema_text,
            "schema_source": schema_meta.get("source"),
            "schema_error": schema_meta.get("error"),
            "schema_canonical": canonical_schema,
            "heuristic_grounding": heuristic,
            "valid_actions": list(valid_actions),
            "is_noop": is_noop,
        }
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
        exp.extras = extras
        # Mirror into metadata so Experience.to_dict() persists it to JSON.
        existing_meta = getattr(exp, "metadata", None) or {}
        if isinstance(existing_meta, dict):
            existing_meta = dict(existing_meta)
        else:
            existing_meta = {}
        existing_meta.update(extras)
        exp.metadata = existing_meta
        experiences.append(exp)

        # Crash-safe per-step persistence: flush this step (frame_path, schema,
        # action, reward, reasoning, raw VLM/LLM responses, valid_actions, etc.)
        # to the streaming JSONL *before* taking the next env step. A kill
        # between here and the next env.step() loses zero step-level data.
        if step_stream_path is not None:
            try:
                step_record = exp.to_dict()
                step_record["env_id"] = env_id
                step_record["episode_index"] = ep_idx
                step_record["step"] = step
                step_record["frame_path"] = img_path
                with open(step_stream_path, "a", encoding="utf-8") as _sf:
                    _sf.write(
                        json.dumps(step_record, ensure_ascii=False, default=str) + "\n"
                    )
                    _sf.flush()
                    try:
                        os.fsync(_sf.fileno())
                    except OSError:
                        pass
            except Exception as _stream_exc:
                logger.warning(
                    "step_stream write failed (env=%s ep=%d step=%d): %s",
                    env_id, ep_idx, step, _stream_exc,
                )

        if verbose:
            r_short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
            tag = " [NOOP]" if is_noop else ""
            print(
                f"  step {step:>3}: action={action!r:<22} "
                f"reward={reward:+.2f} cum={total_reward:+.2f}{tag} "
                f"schema={schema_meta.get('source')} reason={r_short}"
            )

        obs = next_obs
        if done:
            break

    elapsed = time.time() - t0

    try:
        env.close()
    except Exception:
        pass

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="gym_v",
        game_name=display_name,
    )
    episode.set_outcome()

    stats: Dict[str, Any] = {
        "env_id": env_id,
        "display_name": display_name,
        "genre": genre,
        "wrapper": "TemporalVisualGroundingWrapper",
        "macro": False,
        "steps": len(experiences),
        "total_reward": total_reward,
        "terminated": terminated_all,
        "truncated": truncated_all,
        "elapsed_seconds": round(elapsed, 2),
        "model": fallback_model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_gymv",
        "use_vision": use_vision,
        "schema_calls": schema_calls,
        "schema_ok": schema_ok,
        "action_llm_ok": action_llm_ok,
        "action_llm_fail": action_llm_fail,
        "noop_steps": sum(1 for h in history if h["noop"]),
    }
    return episode, stats


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def _count_existing_episodes(env_dir: Path) -> int:
    if not env_dir.exists():
        return 0
    return sum(
        1 for f in env_dir.glob("episode_*.json")
        if f.name != "episode_buffer.json"
    )


def _save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]):
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_env_rollouts(
    env_id: str,
    spec: Any,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    client: Any,
    routed_model: str,
    schema_helpers: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all episodes for one Gym-V env id and persist outputs."""
    safe = _sanitize_env_id(env_id)
    env_dir = output_dir / safe
    env_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = env_dir / "rollouts.jsonl"

    target_episodes = args.episodes
    effective_max_steps = args.max_steps

    start_idx = 0
    if args.resume:
        start_idx = _count_existing_episodes(env_dir)
        if start_idx >= target_episodes:
            print(f"  [SKIP] {env_id}: {start_idx}/{target_episodes} episodes already done")
            return {
                "env_id": env_id, "skipped": True, "existing": start_idx,
                "target_episodes": target_episodes,
            }
        if start_idx > 0:
            print(f"  [RESUME] {env_id}: starting from episode {start_idx}")

    buffer = Episode_Buffer(buffer_size=target_episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t_env = time.time()

    # Per-env streaming dir: one append-only JSONL per episode is enough
    # for crash-safety (we always know which episode was in flight by the
    # mtime). Goes alongside the sealed episode_NNN.json for easy diffing.
    steps_stream_dir = env_dir / "steps_stream"
    steps_stream_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(start_idx, target_episodes):
        print(f"\n  [{env_id}] Episode {ep_idx + 1}/{target_episodes}")
        try:
            frames_dir = (
                env_dir / "frames" / f"ep_{ep_idx:03d}"
                if args.save_frames else None
            )
            step_stream_path = steps_stream_dir / f"ep_{ep_idx:03d}.jsonl"

            episode, stats = run_actor_episode(
                env_id=env_id,
                spec=spec,
                max_steps=effective_max_steps,
                client=client,
                routed_model=routed_model,
                fallback_model=args.model,
                schema_helpers=schema_helpers,
                use_vision=not args.no_vision,
                temperature_action=args.temperature_action,
                temperature_schema=args.temperature_schema,
                frames_dir=frames_dir,
                seed=args.seed_base + ep_idx,
                verbose=args.verbose,
                step_stream_path=step_stream_path,
                ep_idx=ep_idx,
                reasoning_effort=getattr(args, "reasoning_effort", None),
                frame_skip=getattr(args, "frame_skip", DEFAULT_FRAME_SKIP),
            )
            stats["episode_index"] = ep_idx
            print(
                f"    steps={stats['steps']:>3} "
                f"reward={stats['total_reward']:+.2f} "
                f"schema_ok={stats['schema_ok']}/{stats['schema_calls']} "
                f"action_ok={stats['action_llm_ok']} (fail={stats['action_llm_fail']})"
            )

            buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = episode.to_dict()
            ep_data["metadata"] = stats
            with open(env_dir / f"episode_{ep_idx:03d}.json", "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)
            _save_episode_jsonl(episode, jsonl_path, stats)

        except Exception as exc:
            print(f"    [ERROR] episode {ep_idx + 1} failed: {exc}")
            traceback.print_exc()
            all_stats.append({
                "env_id": env_id,
                "episode_index": ep_idx,
                "error": str(exc),
                "steps": 0,
                "total_reward": 0.0,
            })
            continue

    elapsed_env = time.time() - t_env
    buffer.save_to_json(str(env_dir / "episode_buffer.json"))
    print(f"\n  Saved {len(buffer)} episodes for {env_id} in {elapsed_env:.1f}s")

    summary: Dict[str, Any] = {
        "env_id": env_id,
        "display_name": getattr(spec, "display_name", env_id),
        "genre": getattr(spec, "genre", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_gymv",
        "wrapper": "TemporalVisualGroundingWrapper",
        "target_episodes": target_episodes,
        "completed_episodes": len([s for s in all_stats if "error" not in s]),
        "use_vision": not args.no_vision,
        "max_steps": effective_max_steps,
        "frame_skip": getattr(args, "frame_skip", DEFAULT_FRAME_SKIP),
        "elapsed_seconds": round(elapsed_env, 2),
        "episode_stats": all_stats,
    }
    rewards = [s["total_reward"] for s in all_stats if "error" not in s]
    steps_list = [s["steps"] for s in all_stats if "error" not in s]
    if rewards:
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)
    if steps_list:
        summary["mean_steps"] = sum(steps_list) / len(steps_list)

    with open(env_dir / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cold-start actor-agent rollouts using gpt-5.5 visual grounding "
            "+ schema-driven action selection over gym-v Temporal envs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--envs", type=str, nargs="+", default=DEFAULT_ENVS,
        help=(
            "Gym-V Temporal/* env ids to run "
            f"(default: {' '.join(DEFAULT_ENVS)})"
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Episodes per env (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--frame_skip", "--frame-skip",
        type=int, default=DEFAULT_FRAME_SKIP,
        help=(
            "Number of emulator frames each agent action is held for. "
            "Genesis runs at 60 fps, so frame_skip=1 (default for "
            "back-compat) makes a 100-step episode only 1.67 s of real "
            "game time and reward signal is starved on most retro games. "
            "Recommended: 8 (matches standard Atari/Retro RL convention "
            "and gives 100-step episodes ~13 s of game time)."
        ),
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Backbone model for visual grounding + actor (default: {DEFAULT_MODEL})",
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
        "--no_vision", action="store_true",
        help="Skip the vision call; use the deterministic heuristic schema "
             "(or raw text observations) for action selection.",
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="Persist the PNG frames sent to the VLM under <env>/frames/.",
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
        "--output_dir", type=str, default=None,
        help="Output directory (default: <codebase_root>/Cold-start-out-gymv)",
    )
    parser.add_argument(
        "--allow_missing_rom", action="store_true",
        help="Don't pre-skip envs whose ROM stable-retro can't resolve "
             "(useful when the registry is unusual).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-step details (action, reward, schema source).",
    )
    parser.add_argument(
        "--seed_base", type=int, default=42,
        help="Base env seed; per-episode seed = seed_base + ep_idx. "
             "Bumping this lets parallel shard processes use disjoint env "
             "seeds while still numbering their episodes from 0 locally.",
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
            "the visible <state> and action JSON, so hidden thinking is "
            "wasted spend."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    # Resolve env ids against the gymv_wrapper registry.
    try:
        gym_v, TEMPORAL_GAME_SPECS, _ = _import_gymv_stack()
    except Exception as exc:
        print(f"[FATAL] gym_v / gymv_wrapper not importable: {exc}")
        sys.exit(2)

    resolved: List[Tuple[str, Any]] = []
    skipped: List[Tuple[str, str]] = []
    for env_id in args.envs:
        spec = TEMPORAL_GAME_SPECS.get(env_id)
        if spec is None:
            skipped.append((env_id, "unknown env id (not in TEMPORAL_GAME_SPECS)"))
            continue
        if not args.allow_missing_rom and not _rom_resolves(spec.retro_game):
            skipped.append(
                (env_id, f"ROM missing for retro game {spec.retro_game!r}")
            )
            continue
        resolved.append((env_id, spec))

    if not resolved:
        print(
            "[ERROR] No runnable Gym-V Temporal envs. "
            "Check ROM imports for stable-retro, or pass --allow_missing_rom."
        )
        if skipped:
            for env_id, reason in skipped:
                print(f"  - {env_id}: {reason}")
        sys.exit(2)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else CODEBASE_ROOT / "Cold-start-out-gymv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    has_key = bool(
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not has_key:
        print("[WARNING] No API key set. Both VLM and actor calls will fail.")
        print("  Set: export OPENAI_API_KEY='sk-...' or OPENROUTER_API_KEY='sk-or-...'")

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

    print("=" * 78)
    print("  Cold-Start Actor Agent — gym-v Temporal + gpt-5.5")
    print("=" * 78)
    if _API_KEYS_FILE_USED is not None:
        print(f"  API keys file:      {_API_KEYS_FILE_USED}")
    print(f"  Envs:               {', '.join(env_id for env_id, _ in resolved)}")
    if skipped:
        for env_id, reason in skipped:
            print(f"  Skipped:            {env_id} ({reason})")
    print(f"  Episodes (per env): {args.episodes}")
    print(f"  Max steps:          {args.max_steps}")
    print(f"  Frame skip:         {args.frame_skip}")
    print(f"  Model (configured): {args.model}")
    print(f"  Model (routed):     {routed_model}")
    print(f"  Vision schema:      {'OFF (--no_vision)' if args.no_vision else 'ON'}")
    print(f"  Save frames:        {args.save_frames}")
    print(f"  Resume:             {args.resume}")
    print(f"  Output:             {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    env_summaries: List[Dict[str, Any]] = []
    for env_id, spec in resolved:
        print(f"\n{'━' * 78}")
        print(f"  ENV: {env_id} — {getattr(spec, 'display_name', env_id)}")
        print(f"{'━' * 78}")
        summary = run_env_rollouts(
            env_id, spec,
            args=args,
            output_dir=output_dir,
            client=client,
            routed_model=routed_model,
            schema_helpers=schema_helpers,
        )
        env_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor_gymv",
        "use_vision": not args.no_vision,
        "envs": [env_id for env_id, _ in resolved],
        "skipped_envs": [env_id for env_id, _ in skipped],
        "episodes_per_env": args.episodes,
        "max_steps": args.max_steps,
        "frame_skip": args.frame_skip,
        "temperature_action": args.temperature_action,
        "temperature_schema": args.temperature_schema,
        "elapsed_seconds": round(overall_elapsed, 2),
        "per_env_summaries": env_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ACTOR COLD-START (GYM-V) — BATCH COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Envs processed:   {len(env_summaries)}")
    completed = [
        s for s in env_summaries
        if not s.get("skipped") and "completed_episodes" in s
    ]
    total_eps = sum(s["completed_episodes"] for s in completed)
    print(f"  Total episodes:   {total_eps}")
    print(f"  Elapsed:          {overall_elapsed:.1f}s")
    print(f"  Output:           {output_dir}")
    print(f"  Master summary:   {master_path}")
    if completed:
        means = [s["mean_reward"] for s in completed if "mean_reward" in s]
        steps_means = [s["mean_steps"] for s in completed if "mean_steps" in s]
        if means:
            print(f"  Avg reward:       {sum(means) / len(means):.2f}")
        if steps_means:
            print(f"  Avg steps:        {sum(steps_means) / len(steps_means):.1f}")
    print()
    print("  Load into trainer:")
    print("    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records")
    print(f"    eps = load_episodes_from_jsonl('{output_dir}/<env_id_safe>/rollouts.jsonl')")
    print("    records = episodes_to_rollout_records(eps)")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
