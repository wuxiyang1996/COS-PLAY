#!/usr/bin/env python
"""
Cold-start actor-agent rollouts for env_wrappers games (gpt-5.5 vision pipeline).

Pipeline (one outer step):

  1. Render the wrapper's current frame (PIL/np via ``env_wrappers.visual_utils``).
  2. Visual grounding: gpt-5.5 (vision) converts the frame into a canonical
     ``<state>...</state>`` schema using ``vlm_wrapper.schema``.  The wrapper's
     auxiliary text observation is shipped only as supporting context.
  3. Action selection: gpt-5.5 reads the schema + the harness's valid-action
     list and picks ONE action via OpenAI function calling.
  4. ``env.step(action)`` and an :class:`Experience` is appended to the
     :class:`Episode`.  The schema, raw VLM output, action reasoning, and
     reward are all preserved on the Experience so SFT/GRPO consumers can
     replay the trajectory exactly.

Supported env wrappers (see ``env_wrappers/__init__.py``):

  - ``twenty_forty_eight`` / ``candy_crush`` / ``tetris`` via
    ``make_gaming_env(observation_mode="both")`` + ``GamingAgentNLWrapper``.
    **Tetris always goes through** :class:`TetrisMacroActionWrapper` so
    each LLM decision commits one macro-action (rotation + column
    placement) instead of a primitive key press — primitive Tetris is
    not exposed through this script.
  - ``super_mario`` via ``make_orak_env("super_mario", input_modality="text_image")``.

Output layout (``<codebase_root>/Cold-start-out/<game>/``):

  - ``episode_NNN.json``       individual Episode (Episode.to_dict())
  - ``episode_buffer.json``    Episode_Buffer (loadable for trainer)
  - ``rollouts.jsonl``         append-only JSONL, one Episode per line
  - ``rollout_summary.json``   per-game stats
  - ``frames/<ep>/step_NNN.png``  rendered frames sent to the VLM (debug)

Usage (from the Game-AI-Agent root, with GamingAgent / Orak as siblings)::

    export OPENAI_API_KEY="sk-..."          # or OPENROUTER_API_KEY
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$(pwd)/../Orak/src:${PYTHONPATH}"

    # All four games, default 5 episodes each
    python cold_start/generate_cold_start_actor.py

    # 2048 + candy crush, 10 episodes each, max 40 steps
    python cold_start/generate_cold_start_actor.py \\
        --games twenty_forty_eight candy_crush --episodes 10 --max_steps 40

    # Tetris with macro actions, 5 episodes (default for tetris)
    python cold_start/generate_cold_start_actor.py --games tetris

    # Super Mario via Orak (use the dedicated launcher instead — needs
    # the orak-mario conda env + Xvfb).
    python cold_start/generate_cold_start_actor.py --games super_mario --episodes 3 -v

    # Skip the visual grounding stage (uses text obs only, faster, cheaper)
    python cold_start/generate_cold_start_actor.py --no_vision

    # Resume an interrupted run
    python cold_start/generate_cold_start_actor.py --resume
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
# Path setup — make Game-AI-Agent / GamingAgent / Orak importable.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"
ORAK_SRC = CODEBASE_ROOT.parent / "Orak" / "src"

for _p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT), str(ORAK_SRC)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


def _bootstrap_api_keys_from_file() -> Optional[Path]:
    """Seed ``os.environ`` from a sibling ``api_keys.py`` if present.

    Looked-up locations (first hit wins):
      1. ``$COSPLAY_API_KEYS_FILE``
      2. ``cold_start/api_keys.py``
      3. ``<codebase_root>/api_keys.py``
      4. ``<codebase_root>/../api_keys.py``  (workspace root)

    The file should define any of:
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
except Exception:  # pragma: no cover — keep script runnable in isolation
    _SFT_TEACHER_MODEL = "gpt-5.5"

DEFAULT_MODEL = _SFT_TEACHER_MODEL  # gpt-5.5

try:
    from API_func import OPENROUTER_BASE, make_openai_client, effective_openai_model
except Exception:  # pragma: no cover
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    make_openai_client = None
    effective_openai_model = None

logger = logging.getLogger("cold_start.actor")


# ---------------------------------------------------------------------------
# Constants / per-game registry
# ---------------------------------------------------------------------------

GAME_TASK: Dict[str, str] = {
    "twenty_forty_eight": (
        "Play 2048 on a 4x4 grid. Slide tiles up/down/left/right to merge "
        "matching numbers; goal is to create the highest tile possible "
        "(2048 wins). Larger merged tiles score more."
    ),
    "candy_crush": (
        "Match-3 puzzle on an 8x8 colored grid. Swap two adjacent candies "
        "to form lines of 3+ same colors, which clear them and earn points. "
        "Limited number of moves per episode."
    ),
    "tetris": (
        "Classic Tetris. Each macro action commits one piece placement "
        "(rotation + column). Clear lines by filling rows; the game ends "
        "when the stack reaches the top. Prefer placements that clear "
        "lines, avoid creating holes, and keep the stack flat / low."
    ),
    "super_mario": (
        "Super Mario Bros (NES, world 1-1). Move Mario right, jump over "
        "enemies (goombas, koopas) and pits, reach the flag at the end. "
        "Each action is one of 7 jump levels; level 0 = no jump (run), "
        "higher levels = longer/higher jumps."
    ),
}

# Default per-game step caps.
#
# Mirrors the upstream COS-PLAY repo (https://github.com/wuxiyang1996/COS-PLAY)
# verbatim, so trajectories run until a natural win/lose just like the paper:
#
#   - twenty_forty_eight / candy_crush / tetris come from the
#     ``COLD_START_MAX_STEPS_NATURAL_END`` table in
#     ``cold_start/generate_cold_start.py``.
#   - super_mario comes from ``cold_start/run_coldstart_orak_mario.sh``
#     which hard-codes ``--max_steps 100``.
#
# NOTE: Tetris in this script ALWAYS goes through ``TetrisMacroActionWrapper``,
# so one step = one full piece placement (rotation + column drop), whereas
# upstream's 200 is for primitive keypresses. Matching the literal upstream
# value here means episodes can run up to 200 piece placements before
# truncation, which in practice always natural-ends earlier (game-over).
DEFAULT_MAX_STEPS: Dict[str, int] = {
    "twenty_forty_eight": 200,
    "candy_crush": 50,
    "tetris": 200,
    "super_mario": 100,
}

# Default episode count per game when ``--episodes`` is not given.
DEFAULT_EPISODES: Dict[str, int] = {
    "twenty_forty_eight": 5,
    "candy_crush": 5,
    "tetris": 5,
    "super_mario": 3,
}

# Game-name aliases accepted on the CLI (mirrors generate_envwrappers_*).
_GAME_ALIASES: Dict[str, str] = {
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

# Anti-noop: force a different action after this many consecutive steps
# whose state is identical AND reward ≤ 0.
_MAX_CONSECUTIVE_NOOPS = 2
# Number of recent action results to surface in the action-selection prompt.
_HISTORY_WINDOW = 5
# Pixel cap when sending the frame to the VLM (cost control).
_VLM_IMAGE_MAX_SIDE = 1024
# Default token budget for the action call.
# With the strict-enum-only `choose_action` schema (see
# `_build_action_tools`), the tool_call payload is `{"action": "<name>"}`
# — typically 7–14 tokens.  Setting this to 128 gives ~10× safety
# headroom while staying tiny relative to the input prompt: dense
# late-game boards (tetris 20×10 with many filled cells, candy_crush
# 8×8) can reach ~7.9 K input tokens, and the 9B vLLM is served at
# --max-model-len 8192, so anything bigger here trips
# `BadRequestError: This model's maximum context length is 8192`.
# Reasoning models (gpt-5.x, o1/o3/o4) use the `max(6000, max_tokens*4)`
# ceiling in `_chat_completion`, so they remain unaffected.
_ACTION_MAX_TOKENS = 128
# Default token budget for the schema (vision) call. Reasoning models burn
# many internal tokens — this is the *output* cap, not the prompt budget.
# 1500 was too tight for dense boards (candy_crush 8×8, tetris 20×10): the
# response was truncated mid-schema, so the closing ``</state>`` tag never
# appeared and the strict parser dropped to ``fallback_canonical``.
_SCHEMA_MAX_TOKENS = 4000
# Reasoning models (gpt-5.x, o1/o3/o4) charge thinking tokens against the same
# budget, so we hand them an even larger output cap up front.
_SCHEMA_MAX_TOKENS_REASONING = 12000

# Models that require ``max_completion_tokens`` (no ``temperature`` either).
# Matches: gpt-5, gpt-5.5, gpt-5-mini, gpt-5-nano, gpt-5.5-pro, openai/gpt-5*,
# o1, o1-preview, o1-mini, o3, o3-mini, o4, o4-mini.
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


# ---------------------------------------------------------------------------
# Env-wrapper construction
# ---------------------------------------------------------------------------

def _resolve_game_name(name: str) -> Optional[str]:
    """Map a CLI alias to a canonical game key, or None when unknown."""
    return _GAME_ALIASES.get(name.strip().lower())


def _build_env(game: str, max_steps: int):
    """Construct the wrapped env for ``game``.

    Returns ``(env, env_meta)`` where ``env`` exposes
    ``reset()`` -> ``(obs_text, info)`` and
    ``step(action)`` -> ``(obs_text, reward, terminated, truncated, info)``.

    ``info["action_names"]`` is always populated and is the source of truth
    for the action-selection vocabulary.

    Tetris is **always** wrapped with :class:`TetrisMacroActionWrapper`
    so every LLM call commits a full piece placement.
    """
    if game in {"twenty_forty_eight", "candy_crush", "tetris"}:
        from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
        from env_wrappers.gym_like import make_gaming_env

        gym = make_gaming_env(
            game,
            max_steps=max_steps,
            observation_mode="both",                              # text + PNG frame
            render_mode="rgb_array" if game == "tetris" else None,
            load_image_array=True,
        )
        nl_env = GamingAgentNLWrapper(
            gym, include_action_hint=False, game_name=game,
        )

        if game == "tetris":
            from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
            env = TetrisMacroActionWrapper(nl_env)
            wrapper_name = "TetrisMacroActionWrapper"
            macro = True
        else:
            env = nl_env
            wrapper_name = "GamingAgentNLWrapper"
            macro = False

        return env, {
            "wrapper": wrapper_name,
            "underlying": "make_gaming_env",
            "game": game,
            "task": GAME_TASK.get(game, ""),
            "macro": macro,
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
            "game": game,
            "task": GAME_TASK.get(game, ""),
            "macro": False,
        }

    raise ValueError(f"Unsupported game for actor cold-start: {game!r}")


# ---------------------------------------------------------------------------
# Visual extraction helpers
# ---------------------------------------------------------------------------

def _extract_pil_frame(info: Dict[str, Any]):
    """Return a ``PIL.Image`` of the current env frame, or ``None``.

    Tries the cross-env helpers first, falls back to digging into
    ``info["raw_obs"]``.
    """
    try:
        from env_wrappers.visual_utils import get_obs_pil_image
        pil = get_obs_pil_image(info)
        if pil is not None:
            return pil
    except Exception:
        pass

    raw = info.get("raw_obs")
    if isinstance(raw, dict):
        try:
            from env_wrappers.visual_utils import get_obs_pil_image
            pil = get_obs_pil_image(raw)
            if pil is not None:
                return pil
        except Exception:
            pass
        # Manual fallback for Orak / GamingAgent.
        for key in ("frame", "image"):
            arr = raw.get(key)
            if arr is not None:
                try:
                    from PIL import Image
                    a = np.asarray(arr)
                    if a.dtype != np.uint8:
                        a = (a * 255).clip(0, 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
                    if a.ndim == 3 and a.shape[-1] == 4:
                        a = a[..., :3]
                    return Image.fromarray(a, mode="RGB")
                except Exception:
                    continue
    return None


def _save_frame(pil, path: Path) -> Optional[str]:
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
    """Return ``(client, routed_model)`` or ``(None, model)`` on failure.

    Mirrors the helper used in ``visual_grounding_tests/`` so the actor
    transparently routes through OpenRouter when ``OPENROUTER_API_KEY`` is
    set, otherwise direct OpenAI.
    """
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


def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: Optional[list] = None,
    tool_choice: Any = None,
):
    """Cross-model chat-completion wrapper.

    Reasoning models (``gpt-5.x``, ``o1``/``o3``/``o4``) reject ``max_tokens``
    and ``temperature`` and require ``max_completion_tokens`` — and they spend
    much of that budget on hidden thinking tokens, so the visible output gets
    truncated when the cap is too low.

    We detect reasoning models up front and route them straight through with
    a generous ``max_completion_tokens``.  Non-reasoning models keep the
    legacy ``max_tokens``/``temperature`` path, with a single fallback retry
    if the server unexpectedly demands the reasoning-style fields.
    """
    if _is_reasoning_model(model):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            # Reasoning + visible output share the same budget; without a
            # generous cap, dense boards (candy_crush 8×8, tetris 20×10)
            # truncate mid-schema and the parser falls back to canonical.
            "max_completion_tokens": max(6000, max_tokens * 4),
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
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

    # Thinking-mode-class models (Qwen3*, Qwen3.5*, DeepSeek-R1-distill,
    # qwen3.5-plus / qwen3.6-* on DashScope/OpenRouter, etc.) emit a
    # free-form `<think>...</think>` block before the tool call AND, when
    # served by Alibaba DashScope (or proxied through OpenRouter), reject
    # strict `tool_choice={"type":"function",...}` payloads with HTTP 400
    # ("InvalidParameter ... in thinking mode").  We disable thinking via
    # *both* recognised parameter names so the same payload works whether
    # the endpoint is local-vLLM or DashScope/OpenRouter:
    #
    #   - vLLM-OpenAI-compatible: ``extra_body.chat_template_kwargs
    #     .enable_thinking=False`` is forwarded to the Jinja chat template
    #     so no ``<think>`` block is emitted.
    #   - DashScope (and OpenRouter when it proxies Alibaba upstream):
    #     ``extra_body.enable_thinking=False`` at the root, per Alibaba's
    #     OpenAI-compat spec.  Each server silently ignores the key it
    #     does not recognise.
    #
    # Heuristic: model id contains a slash (HuggingFace ``<org>/<name>``
    # for vLLM **or** OpenRouter ``<provider>/<slug>``).  Managed APIs
    # like ``gpt-4o`` / ``claude-3.5-sonnet`` / ``o3-mini`` do not.  This
    # pairs with the strict `enum`-only action tool schema (see
    # ``_build_action_tools``) to keep tool_call output tightly bounded
    # (~10 tokens), well below the cap.
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
        kwargs["max_completion_tokens"] = max(6000, max_tokens * 5)
        return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Stage 1 — visual schema generation (gpt-5.5 vision)
# ---------------------------------------------------------------------------

def _import_schema_helpers():
    """Lazy import of ``vlm_wrapper.schema`` helpers (optional dep)."""
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
    """Salvage a ``<state>...</state>`` schema from messy VLM output.

    Returns ``(parsed_schema_or_None, recovery_kind)``.  ``recovery_kind`` is
    one of:

    - ``"strict"``       — the upstream strict parser succeeded
    - ``"fenced"``       — markdown fences were stripped before re-parsing
    - ``"truncated"``    — ``<state>`` opened but ``</state>`` was missing
                           (response cut off — we appended the closing tag)
    - ``"untagged"``     — sections like ``<entities>`` present without
                           an enclosing ``<state>`` tag — we wrapped them
    - ``""``             — nothing recoverable
    """
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


def _import_canonical_helpers():
    """Lazy import of canonical-schema helpers (optional dep)."""
    try:
        from visual_grounding_tests.canonical_schema import (
            MAX_ENTITIES_BY_GAME,
            canonical_label_hint,
            make_canonical_schema,
        )
        return {
            "max_entities": MAX_ENTITIES_BY_GAME,
            "canonical_label_hint": canonical_label_hint,
            "make_canonical_schema": make_canonical_schema,
        }
    except Exception as exc:
        logger.debug("canonical_schema unavailable: %s", exc)
        return None


def generate_schema_from_image(
    *,
    pil_image,
    obs_text: str,
    game: str,
    task_id: str,
    goal: str,
    step: int,
    valid_actions: List[str],
    client: Any,
    routed_model: str,
    schema_helpers: Dict[str, Any],
    canonical_helpers: Optional[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = _SCHEMA_MAX_TOKENS,
    canonical_fallback: Optional[str] = None,
) -> Dict[str, Any]:
    """Call gpt-5.5 (vision) to produce a ``<state>...</state>`` schema.

    The image is the primary input; ``obs_text`` rides along as auxiliary
    context. Returns a dict with the parsed ``schema`` (or ``None``), the
    raw model output, the routed model id, and any exception captured.

    When the API call fails or no schema is parsed, falls back to the
    deterministic ``canonical_fallback`` if provided (so the actor can
    keep going even when the VLM is unreachable).
    """
    if pil_image is None or schema_helpers is None or client is None:
        return {
            "schema": canonical_fallback,
            "raw": "",
            "source": "fallback_canonical" if canonical_fallback else "no_image_or_client",
            "error": None,
        }

    canonical_hint = None
    max_entities = 20
    if canonical_helpers is not None:
        try:
            canonical_hint = canonical_helpers["canonical_label_hint"](game)
        except Exception:
            canonical_hint = None
        max_entities = (canonical_helpers["max_entities"] or {}).get(game, 20)

    system = schema_helpers["build_system_prompt"]("gymv", max_entities=max_entities)
    if canonical_hint:
        system = system + "\n\n" + canonical_hint

    extra_parts: List[str] = [f"Game rules:\n{GAME_TASK.get(game, '') or task_id}"]
    if obs_text:
        extra_parts.append(
            "Environment text state (auxiliary — for reference only):\n"
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
        logger.warning("[schema-VLM] %s step %d failed: %s", game, step, exc)

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
                game, step, recovery, finish_reason, len(raw),
            )
        return {"schema": parsed, "source": "vlm", **base_meta}

    if raw and finish_reason == "length":
        logger.warning(
            "[schema-VLM] %s step %d response truncated (finish_reason=length, raw_len=%d) — "
            "consider raising _SCHEMA_MAX_TOKENS",
            game, step, len(raw),
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
    "You are an Actor Agent for the COS-PLAY game-AI pipeline.\n"
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

    The schema is intentionally minimal — just the ``action`` field with
    a strict ``enum`` — so the model cannot generate free-form text that
    blows past the token budget.

    Earlier iterations carried an optional ``reasoning`` (chain-of-thought)
    field, but on vLLM-served thinking models (Qwen3 / Qwen3.5,
    DeepSeek-R1-distill, …) the model would write multi-paragraph
    reasoning into that field — exceeding ``_ACTION_MAX_TOKENS`` and
    truncating the JSON mid-string ("Unterminated string starting at:
    line N column M").  Reasoning was only ever used for log diagnostics,
    not for downstream metrics or training data, so dropping it is the
    cleaner fix.

    Backends that perform constrained / guided generation (vLLM with
    ``--enable-auto-tool-choice --tool-call-parser hermes`` plus the
    default outlines/lm-format-enforcer backend) honor the ``enum`` and
    refuse to emit tokens outside the valid set.  Backends that don't
    (raw OpenAI, Anthropic) still get the description as a hint and the
    actor's downstream ``_canonicalize_action`` accepts any reasonable
    spelling.
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

    # Numbered selection ("1", "1.", "3)").
    m = re.match(r"^\s*(\d+)\s*[\.\)\-:]?\s*$", cand)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(action_names):
            return action_names[idx]

    # Prefix / substring fallback.
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
    game: str,
    step: int,
    history: List[Dict[str, Any]],
    client: Any,
    routed_model: str,
    temperature: float = 0.4,
    max_tokens: int = _ACTION_MAX_TOKENS,
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Call gpt-5.5 with the schema → ``(action, reasoning, raw, error)``.

    ``action`` is canonicalised against ``valid_actions``; returns
    ``None`` for ``action`` only when the call fails AND no fallback can
    be applied (caller will pick a deterministic default in that case).
    """
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
        f"Game: {game}",
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

        # No tool call — try to extract an action from the message content.
        canonical = _canonicalize_action(raw, valid_actions)
        if canonical:
            return canonical, None, raw, None

    except Exception as exc:
        err = repr(exc)
        logger.warning("[action-LLM] %s step %d failed: %s", game, step, exc)

    return None, None, raw, err


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _is_noop(prev_obs: str, next_obs: str, reward: float) -> bool:
    """Best-effort no-op detection: state unchanged AND non-positive reward."""
    if reward and float(reward) > 0.0:
        return False
    return (prev_obs or "") == (next_obs or "")


def _pick_different(action: str, candidates: List[str]) -> str:
    alts = [a for a in candidates if a != action]
    return random.choice(alts) if alts else action


def run_actor_episode(
    *,
    env: Any,
    env_meta: Dict[str, Any],
    game: str,
    max_steps: int,
    client: Any,
    routed_model: str,
    fallback_model: str,
    schema_helpers: Optional[Dict[str, Any]],
    canonical_helpers: Optional[Dict[str, Any]],
    use_vision: bool,
    temperature_action: float,
    temperature_schema: float,
    frames_dir: Optional[Path],
    seed: Optional[int],
    verbose: bool,
    step_stream_path: Optional[Path] = None,
    ep_idx: int = 0,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode end-to-end and return ``(Episode, stats)``.

    If ``step_stream_path`` is provided, every completed step is flushed to
    that path as a single JSON line *immediately* after the env step. This
    makes the rollout crash-safe: a SIGTERM / OOM / API outage mid-episode
    will preserve every step that finished before the failure. The stream is
    truncated at episode start so a re-run via ``--resume`` cannot mix old
    partial data into a fresh attempt.
    """
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

    task = GAME_TASK.get(game, env_meta.get("task", ""))
    task_id = f"{env_meta.get('underlying', 'env')}/{game}"
    goal = task.split("\n")[0] if task else game

    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    info = dict(info or {})

    experiences: List[Experience] = []
    history: List[Dict[str, Any]] = []
    consecutive_noops = 0
    last_noop_action: Optional[str] = None
    schema_calls = 0
    schema_ok = 0
    action_llm_ok = 0
    action_llm_fail = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    t0 = time.time()
    for step in range(max_steps):
        valid_actions: List[str] = list(
            info.get("action_names")
            or info.get("available_actions")
            or env_meta.get("action_names")
            or []
        )[:25]
        if not valid_actions:
            logger.warning("[%s] step %d: empty action vocab — stopping", game, step)
            break

        # 1. Pull a frame (PIL) for the VLM.
        pil = _extract_pil_frame(info)
        img_path: Optional[str] = None
        if pil is not None and frames_dir is not None:
            img_path = _save_frame(pil, frames_dir / f"step_{step:03d}.png")

        # 2. Canonical schema fallback (deterministic, modality-invariant).
        canonical_schema: Optional[str] = None
        if canonical_helpers is not None:
            try:
                canonical_schema = canonical_helpers["make_canonical_schema"](
                    game=game,
                    info=info,
                    task_id=task_id,
                    goal=goal,
                    step=step,
                    actions=valid_actions,
                )
            except Exception as exc:
                logger.debug("canonical_schema(%s) failed: %s", game, exc)

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
                obs_text=obs,
                game=game,
                task_id=task_id,
                goal=goal,
                step=step,
                valid_actions=valid_actions,
                client=client,
                routed_model=routed_model,
                schema_helpers=schema_helpers,
                canonical_helpers=canonical_helpers,
                temperature=temperature_schema,
                max_tokens=schema_budget,
                canonical_fallback=canonical_schema,
            )
            schema_calls += 1
            if schema_meta.get("source") == "vlm":
                schema_ok += 1
            schema_text = schema_meta.get("schema")
        else:
            # No vision: use the canonical schema (or text-only fallback).
            schema_text = canonical_schema
            schema_meta = {
                "schema": canonical_schema,
                "raw": "",
                "source": "canonical" if canonical_schema else "text_only",
                "error": None,
            }

        # 4. Action selection (text-only call: schema → action).
        action, reasoning, action_raw, action_err = select_action_from_schema(
            schema_text=schema_text,
            obs_text=obs,
            valid_actions=valid_actions,
            task=task,
            game=game,
            step=step,
            history=history,
            client=client,
            routed_model=routed_model,
            temperature=temperature_action,
        )
        if action is not None:
            action_llm_ok += 1
        else:
            action_llm_fail += 1
            # Deterministic fallback: pick the first valid action (or a
            # different one if it's a known no-op).
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

        # 6. Step the env.
        try:
            next_obs, reward, terminated, truncated, next_info = env.step(action)
        except Exception as exc:
            logger.error("[%s] step %d env.step(%r) failed: %s", game, step, action, exc)
            if verbose:
                traceback.print_exc()
            break
        next_info = dict(next_info or {})
        reward = float(reward or 0.0)
        total_reward += reward
        done = bool(terminated) or bool(truncated)

        is_noop = _is_noop(obs, next_obs, reward)
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
            state=obs,
            action=str(action),
            reward=reward,
            next_state=next_obs,
            done=done,
            intentions=reasoning,
            tasks=task,
        )
        exp.idx = step
        exp.action_type = "macro" if env_meta.get("macro") else "primitive"
        # Persist raw observation snapshots for replay (capped to keep JSON size sane).
        raw_obs = info.get("raw_obs")
        next_raw_obs = next_info.get("raw_obs")
        exp.raw_state = (str(raw_obs)[:4000] if raw_obs is not None else None)
        exp.raw_next_state = (str(next_raw_obs)[:4000] if next_raw_obs is not None else None)
        exp.available_actions = list(valid_actions)
        exp.interface = {
            "env_name": env_meta.get("underlying") or "env",
            "game_name": game,
            "wrapper": env_meta.get("wrapper"),
        }
        # Stash the schema + VLM outputs on the Experience.extras dict so
        # downstream skill / SFT consumers can replay the trajectory.
        extras = getattr(exp, "extras", None) or {}
        extras["schema"] = schema_text
        extras["schema_source"] = schema_meta.get("source")
        extras["schema_error"] = schema_meta.get("error")
        extras["schema_canonical"] = canonical_schema
        if schema_meta.get("finish_reason") is not None:
            extras["schema_finish_reason"] = schema_meta.get("finish_reason")
        if schema_meta.get("recovery"):
            extras["schema_recovery"] = schema_meta.get("recovery")
        schema_raw = schema_meta.get("raw") or ""
        if schema_raw:
            # Truncate raw VLM output to keep episode JSON files reasonably
            # sized while still enabling post-hoc debugging of parse failures.
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
        extras["valid_actions"] = list(valid_actions)
        extras["is_noop"] = is_noop
        exp.extras = extras
        # Mirror into metadata so Experience.to_dict() persists it to JSON
        # (Experience.to_dict serialises only metadata — extras would be lost).
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
                step_record["game"] = game
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
                    "step_stream write failed (game=%s ep=%d step=%d): %s",
                    game, ep_idx, step, _stream_exc,
                )

        if verbose:
            r_short = (reasoning[:80] + "...") if reasoning and len(reasoning) > 80 else reasoning
            tag = " [NOOP]" if is_noop else ""
            print(
                f"  step {step:>3}: action={action!r:<28} "
                f"reward={reward:+.2f} cum={total_reward:+.2f}{tag} "
                f"schema={schema_meta.get('source')} reason={r_short}"
            )

        obs = next_obs
        info = next_info
        if done:
            break

    elapsed = time.time() - t0

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name=env_meta.get("underlying") or "env",
        game_name=game,
    )
    episode.set_outcome()

    stats: Dict[str, Any] = {
        "game": game,
        "wrapper": env_meta.get("wrapper"),
        "macro": env_meta.get("macro", False),
        "steps": len(experiences),
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "elapsed_seconds": round(elapsed, 2),
        "model": fallback_model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor",
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

def _count_existing_episodes(game_dir: Path) -> int:
    if not game_dir.exists():
        return 0
    return sum(
        1 for f in game_dir.glob("episode_*.json")
        if f.name != "episode_buffer.json"
    )


def _save_episode_jsonl(episode: Episode, jsonl_path: Path, stats: Dict[str, Any]):
    record = episode.to_dict()
    record["rollout_metadata"] = stats
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_game_rollouts(
    game: str,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    client: Any,
    routed_model: str,
    schema_helpers: Optional[Dict[str, Any]],
    canonical_helpers: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all episodes for one game and persist outputs."""
    game_dir = output_dir / game
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    target_episodes = (
        args.episodes if args.episodes is not None
        else DEFAULT_EPISODES.get(game, 5)
    )
    effective_max_steps = (
        args.max_steps if args.max_steps is not None
        else DEFAULT_MAX_STEPS.get(game, 60)
    )

    start_idx = 0
    if args.resume:
        start_idx = _count_existing_episodes(game_dir)
        if start_idx >= target_episodes:
            print(f"  [SKIP] {game}: {start_idx}/{target_episodes} episodes already done")
            return {
                "game": game, "skipped": True, "existing": start_idx,
                "target_episodes": target_episodes,
            }
        if start_idx > 0:
            print(f"  [RESUME] {game}: starting from episode {start_idx}")

    buffer = Episode_Buffer(buffer_size=target_episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t_game = time.time()

    # Per-game streaming dir: one append-only JSONL per episode is enough
    # for crash-safety (we always know which episode was in flight by mtime).
    # Goes alongside the sealed episode_NNN.json for easy diffing.
    steps_stream_dir = game_dir / "steps_stream"
    steps_stream_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(start_idx, target_episodes):
        print(f"\n  [{game}] Episode {ep_idx + 1}/{target_episodes}")
        try:
            env, env_meta = _build_env(
                game,
                max_steps=effective_max_steps,
            )
            frames_dir = (
                game_dir / "frames" / f"ep_{ep_idx:03d}" if args.save_frames else None
            )
            step_stream_path = steps_stream_dir / f"ep_{ep_idx:03d}.jsonl"

            episode, stats = run_actor_episode(
                env=env,
                env_meta=env_meta,
                game=game,
                max_steps=effective_max_steps,
                client=client,
                routed_model=routed_model,
                fallback_model=args.model,
                schema_helpers=schema_helpers,
                canonical_helpers=canonical_helpers,
                use_vision=not args.no_vision,
                temperature_action=args.temperature_action,
                temperature_schema=args.temperature_schema,
                frames_dir=frames_dir,
                seed=args.seed_base + ep_idx,
                verbose=args.verbose,
                step_stream_path=step_stream_path,
                ep_idx=ep_idx,
            )
            try:
                env.close()
            except Exception:
                pass

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
            with open(game_dir / f"episode_{ep_idx:03d}.json", "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)
            _save_episode_jsonl(episode, jsonl_path, stats)

        except Exception as exc:
            print(f"    [ERROR] episode {ep_idx + 1} failed: {exc}")
            traceback.print_exc()
            all_stats.append({
                "game": game,
                "episode_index": ep_idx,
                "error": str(exc),
                "steps": 0,
                "total_reward": 0.0,
            })
            continue

    elapsed_game = time.time() - t_game

    buffer.save_to_json(str(game_dir / "episode_buffer.json"))
    print(f"\n  Saved {len(buffer)} episodes for {game} in {elapsed_game:.1f}s")

    summary: Dict[str, Any] = {
        "game": game,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor",
        "wrapper": "varies",
        "target_episodes": target_episodes,
        "completed_episodes": len([s for s in all_stats if "error" not in s]),
        "use_vision": not args.no_vision,
        "max_steps": effective_max_steps,
        "elapsed_seconds": round(elapsed_game, 2),
        "episode_stats": all_stats,
    }
    rewards = [s["total_reward"] for s in all_stats if "error" not in s]
    steps = [s["steps"] for s in all_stats if "error" not in s]
    if rewards:
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)
    if steps:
        summary["mean_steps"] = sum(steps) / len(steps)

    with open(game_dir / "rollout_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_games(requested: List[str]) -> Tuple[List[str], List[str]]:
    canonical = list(GAME_TASK.keys())
    out: List[str] = []
    skipped: List[str] = []
    for g in requested:
        canon = _resolve_game_name(g)
        if canon and canon in canonical:
            if canon not in out:
                out.append(canon)
        else:
            skipped.append(g)
    return out, skipped


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cold-start actor-agent rollouts using gpt-5.5 visual grounding "
            "+ schema-driven action selection over env_wrappers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--games", type=str, nargs="+", default=list(GAME_TASK.keys()),
        help="Games to run (default: all four). Aliases: 2048, mario, etc.",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Episodes per game (default: per-game DEFAULT_EPISODES)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Max steps per episode (default: per-game DEFAULT_MAX_STEPS)",
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
        help="Skip the vision call; use the deterministic canonical schema "
             "(or raw text observations) for action selection.",
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="Persist the PNG frames sent to the VLM under <game>/frames/.",
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
        help="Output directory (default: <codebase_root>/Cold-start-out)",
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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    games, skipped = _resolve_games(args.games)
    if not games:
        print(
            "[ERROR] No valid games selected. Available: "
            + ", ".join(GAME_TASK.keys())
        )
        sys.exit(2)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else CODEBASE_ROOT / "Cold-start-out"
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
    canonical_helpers = _import_canonical_helpers()

    print("=" * 78)
    print("  Cold-Start Actor Agent — env_wrappers + gpt-5.5")
    print("=" * 78)
    if _API_KEYS_FILE_USED is not None:
        print(f"  API keys file:      {_API_KEYS_FILE_USED}")
    print(f"  Games:              {', '.join(games)}")
    if skipped:
        print(f"  Skipped (unknown):  {', '.join(skipped)}")
    print(f"  Episodes (per game):{args.episodes if args.episodes is not None else '<per-game default>'}")
    print(f"  Max steps:          {args.max_steps if args.max_steps is not None else '<per-game default>'}")
    print(f"  Model (configured): {args.model}")
    print(f"  Model (routed):     {routed_model}")
    print(f"  Vision schema:      {'OFF (--no_vision)' if args.no_vision else 'ON'}")
    print(f"  Macro tetris:       ON (always — TetrisMacroActionWrapper)")
    print(f"  Save frames:        {args.save_frames}")
    print(f"  Resume:             {args.resume}")
    print(f"  Output:             {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []
    for game in games:
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game}")
        print(f"{'━' * 78}")
        summary = run_game_rollouts(
            game,
            args=args,
            output_dir=output_dir,
            client=client,
            routed_model=routed_model,
            schema_helpers=schema_helpers,
            canonical_helpers=canonical_helpers,
        )
        game_summaries.append(summary)

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "agent_type": "vlm_actor",
        "use_vision": not args.no_vision,
        "macro_tetris": True,
        "games": games,
        "skipped_games": skipped,
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "temperature_action": args.temperature_action,
        "temperature_schema": args.temperature_schema,
        "elapsed_seconds": round(overall_elapsed, 2),
        "per_game_summaries": game_summaries,
    }
    master_path = output_dir / "batch_rollout_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  ACTOR COLD-START — BATCH COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Games processed:  {len(game_summaries)}")
    completed = [
        s for s in game_summaries
        if not s.get("skipped") and "completed_episodes" in s
    ]
    total_eps = sum(s["completed_episodes"] for s in completed)
    print(f"  Total episodes:   {total_eps}")
    print(f"  Elapsed:          {overall_elapsed:.1f}s")
    print(f"  Output:           {output_dir}")
    print(f"  Master summary:   {master_path}")
    if completed:
        avg_reward = sum(s.get("mean_reward", 0.0) for s in completed if "mean_reward" in s) / max(
            1, sum(1 for s in completed if "mean_reward" in s)
        )
        avg_steps = sum(s.get("mean_steps", 0.0) for s in completed if "mean_steps" in s) / max(
            1, sum(1 for s in completed if "mean_steps" in s)
        )
        print(f"  Avg reward:       {avg_reward:.2f}")
        print(f"  Avg steps:        {avg_steps:.1f}")
    print()
    print("  Load into trainer:")
    print("    from cold_start.load_rollouts import load_episodes_from_jsonl, episodes_to_rollout_records")
    print(f"    eps = load_episodes_from_jsonl('{output_dir}/<game>/rollouts.jsonl')")
    print("    records = episodes_to_rollout_records(eps)")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
