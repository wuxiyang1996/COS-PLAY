#!/usr/bin/env python
"""
Cold-start actor-agent rollouts for the four visual-reasoning benchmarks.

Pipeline (one outer "step" = one benchmark sample):

  1. Load a sample from one of the four benchmarks declared in
     ``visual_reasoning_wrapper.benchmarks`` (visual_toolbench / tir_bench
     for image QA; video_holmes / siv_bench for video MCQ).
  2. Visual grounding: gpt-5.5 (vision) converts the image — or the
     uniformly-sampled video frames — into a canonical ``<state>...</state>``
     schema using ``vlm_wrapper.schema``.  The benchmark question rides
     along as auxiliary context.
  3. Actor agent: gpt-5.5 reads the schema + the question + the valid
     action space and picks ONE answer via OpenAI function calling:
       - video MCQ benchmarks  -> single letter A..E (SIV) / A..F (Holmes)
       - image QA benchmarks   -> short free-form ``answer`` string
  4. Each sample is persisted with: the schema, the raw VLM response,
     the actor's reasoning + answer, the gold answer, and a string-match
     ``correct`` flag (diagnostic only — official scoring uses rubrics
     for the image benchmarks).

Output layout (default ``<codebase_root>/Cold-start-out-visual-reasoning/<run_id>/``):

  <benchmark>/
      sample_000.json            individual record
      ...
      samples.jsonl              append-only JSONL, one record per sample
      summary.json               per-benchmark stats
      frames/sample_NNN/         saved frames sent to the VLM
                                 (frame_KK.png for video; frame_00.png for image)
  batch_summary.json             top-level summary across the four benchmarks

Usage (from the Multi-hop-Reasoning-VLM-Agent root)::

    # All four benchmarks, 5 test cases each (the default)
    python cold_start/generate_cold_start_actor_visual_reasoning.py

    # Just VTB and TIR-Bench, 3 cases each, save frames
    python cold_start/generate_cold_start_actor_visual_reasoning.py \\
        --benchmarks visual_toolbench tir_bench --num_test_cases 3 -v

    # Skip the vision call (no API spend on the schema stage)
    python cold_start/generate_cold_start_actor_visual_reasoning.py \\
        --no_vision --num_test_cases 2 -v

API keys are auto-loaded from ``<workspace>/api_keys.py`` (or
``$COSPLAY_API_KEYS_FILE``) at import time, identically to the existing
``generate_cold_start_actor.py`` family.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Path setup — make the codebase importable when run as a plain script.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = CODEBASE_ROOT.parent

for _p in [str(CODEBASE_ROOT), str(WORKSPACE_ROOT)]:
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


def _bootstrap_api_keys_from_file() -> Optional[Path]:
    """Seed ``os.environ`` from a sibling ``api_keys.py`` if present.

    Mirrors ``generate_cold_start_actor._bootstrap_api_keys_from_file``.
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
# Imports that need PYTHONPATH / api keys to be set first.
# ---------------------------------------------------------------------------
import openai  # noqa: E402

try:
    from common.models import BACKBONE_SFT_TEACHER_MODEL as _SFT_TEACHER_MODEL
except Exception:  # pragma: no cover — keep script runnable in isolation
    _SFT_TEACHER_MODEL = "gpt-5.5"

DEFAULT_MODEL = _SFT_TEACHER_MODEL  # gpt-5.5

try:
    from API_func import make_openai_client, effective_openai_model
except Exception:  # pragma: no cover
    make_openai_client = None
    effective_openai_model = None

logger = logging.getLogger("cold_start.actor_visual_reasoning")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARKS = ("visual_toolbench", "tir_bench", "video_holmes", "siv_bench")
DEFAULT_NUM_TEST_CASES = 5
# Number of frames to uniformly sample from each video clip when the
# benchmark is a video benchmark.  Mirrors the upstream parsers.
DEFAULT_NUM_FRAMES = 6
# Pixel cap when sending images / video frames to the VLM (cost control).
_VLM_IMAGE_MAX_SIDE = 1024
# Token budgets for the action call.
_ACTION_MAX_TOKENS = 350
# Token budget for the schema (vision) call.  Reasoning models also
# spend hidden tokens against this cap, so we hand them a bigger one.
_SCHEMA_MAX_TOKENS = 4000
_SCHEMA_MAX_TOKENS_REASONING = 12000

# Models that require ``max_completion_tokens`` (no ``temperature``).
# Matches gpt-5.x, gpt-5.5*, o1/o3/o4 families.
_REASONING_MODEL_RE = re.compile(
    r"(?:^|/)(?:gpt-5(?:[\.\-]\w+)?|o[134](?:[\.\-]\w+)?)(?:$|[^\w])",
    re.IGNORECASE,
)

_LETTERS_HOLMES = ("A", "B", "C", "D", "E", "F")
# SIV-Bench rows ship with anywhere from 4 up to 12 options (the
# Relation-Inference sub-task uses A..L); keep the full span so we
# don't silently drop the gold letter on wide-option rows.
_LETTERS_SIV = (
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
)
_LETTERS_ALL = (
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
)

# Detect MCQ-style prompts where each option is rendered as ``A: text``,
# ``A. text``, ``A) text`` (case-insensitive, anchored at line start OR
# preceded by whitespace).  Used by the TIR-Bench / VTB iterators where
# the upstream loaders don't expose an explicit options dict.
_MCQ_OPTION_RE = re.compile(
    r"(?:^|\n|\s)([A-J])\s*[:.\)]\s+(\S[^\n]*)",
    re.MULTILINE,
)


def _detect_inline_mcq(prompt: str, gold: Optional[str]) -> Tuple[bool, List[str], Optional[str]]:
    """If ``prompt`` looks like an inline-MCQ and ``gold`` is a single
    letter, return ``(True, letters, options_block)``.  Otherwise
    ``(False, [], None)``.

    A prompt qualifies when at least two distinct letters out of A..J
    appear with a ``A: …`` / ``A. …`` / ``A) …`` style option marker
    AND the gold answer (if present) is a single letter.
    """
    if not prompt:
        return False, [], None
    if gold is not None:
        g = str(gold).strip()
        if not (len(g) == 1 and g.upper() in _LETTERS_ALL):
            return False, [], None
    matches = _MCQ_OPTION_RE.findall(prompt)
    seen: List[str] = []
    options: Dict[str, str] = {}
    for letter, body in matches:
        L = letter.upper()
        if L not in seen:
            seen.append(L)
        options.setdefault(L, body.strip())
    if len(seen) < 2:
        return False, [], None
    # Preserve canonical letter order.
    ordered = [L for L in _LETTERS_ALL if L in seen]
    return True, ordered, _format_options_block({L: options[L] for L in ordered})


def _is_reasoning_model(model: str) -> bool:
    """Return True for OpenAI-style reasoning models (gpt-5.x, o1/o3/o4)."""
    if not model:
        return False
    return bool(_REASONING_MODEL_RE.search(model))


# ---------------------------------------------------------------------------
# Sample-id filter — when ``--sample_ids_dir <dir>`` is set, every benchmark
# is restricted to the ids listed in ``<dir>/<benchmark>_*.txt`` (one per
# line, lines starting with ``#`` are comments). Manifests are produced by
# ``cold_start/task_samples/build_visual_reasoning_diverse_1000.py``.
# ---------------------------------------------------------------------------

def _load_sample_id_filter(
    sample_ids_dir: Optional[str], benchmark: str
) -> Optional[set]:
    if not sample_ids_dir:
        return None
    base = Path(sample_ids_dir)
    if not base.is_dir():
        logger.warning("sample_ids_dir %s missing — no filter applied", base)
        return None
    # Accept ``<benchmark>.txt`` or ``<benchmark>_*.txt``.
    candidates = sorted(base.glob(f"{benchmark}_*.txt")) + sorted(
        base.glob(f"{benchmark}.txt")
    )
    if not candidates:
        logger.info("no manifest for %s under %s — running unfiltered",
                    benchmark, base)
        return None
    ids: set = set()
    for p in candidates:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    logger.info("[%s] sample_ids_filter: %d ids loaded from %s",
                benchmark, len(ids), candidates[0].name)
    return ids


# ---------------------------------------------------------------------------
# Schema helpers (lazy imports — keep the module importable even if the
# downstream visual stack isn't fully installed).
# ---------------------------------------------------------------------------

def _import_schema_helpers() -> Optional[Dict[str, Any]]:
    try:
        from vlm_wrapper.schema import (
            build_system_prompt,
            build_user_message,
            encode_image_b64,
            parse_schema_output,
        )
        return {
            "build_system_prompt": build_system_prompt,
            "build_user_message": build_user_message,
            "encode_image_b64": encode_image_b64,
            "parse_schema_output": parse_schema_output,
        }
    except Exception as exc:
        logger.warning("vlm_wrapper.schema unavailable: %s", exc)
        return None


# Lenient parser shamelessly copied from generate_cold_start_actor.py — same
# defence-in-depth strategy for salvaging truncated / fenced VLM output.
_LENIENT_STATE_OPEN_RE = re.compile(r"<state\b[^>]*>", re.IGNORECASE)
_LENIENT_STATE_CLOSE_RE = re.compile(r"</state\s*>", re.IGNORECASE)
_LENIENT_FENCE_RE = re.compile(
    r"^\s*```(?:xml|html|text|state)?\s*\n?|\n?```\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_LENIENT_SECTION_RE = re.compile(
    r"<(entities|attributes|affordances|relations|state_flags|targets|actions|evidence|answer)\b",
    re.IGNORECASE,
)


def _lenient_parse_schema(raw: str, strict_parser) -> Tuple[Optional[str], str]:
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


# ---------------------------------------------------------------------------
# Image / video helpers
# ---------------------------------------------------------------------------

def _resize_pil(pil, max_side: int = _VLM_IMAGE_MAX_SIDE):
    """Best-effort resize (longest edge -> max_side)."""
    try:
        from PIL import Image
    except ImportError:
        return pil
    if pil is None:
        return None
    if not hasattr(pil, "size"):
        return pil
    w, h = pil.size
    if max_side <= 0 or max(w, h) <= max_side:
        return pil
    scale = max_side / max(w, h)
    try:
        return pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    except Exception:
        return pil


def _save_pil(pil, path: Path) -> Optional[str]:
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

    Mirrors the helper used by the other cold_start actor scripts so the
    pipeline transparently routes through OpenRouter when
    ``OPENROUTER_API_KEY`` is set, otherwise direct OpenAI.
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
    """Cross-model chat-completion wrapper (handles gpt-5.x reasoning models).

    ``reasoning_effort`` ∈ {minimal, low, medium, high} is forwarded only
    for reasoning models; ignored otherwise.  Visual MCQ benchmarks
    benefit more from hidden chain-of-thought than the env pipelines
    (multi-hop social-causal inference, tool-use composition) — set
    ``medium`` if teacher answer correctness is the bottleneck;
    ``minimal`` if you've measured equal accuracy at lower spend.
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
# Stage 1 — visual schema generation (gpt-5.5 vision; image OR video frames)
# ---------------------------------------------------------------------------

def _build_video_user_message(
    frames: Sequence[Any],
    *,
    schema_helpers: Dict[str, Any],
    domain: str,
    task_id: str,
    goal: str,
    extra_context: str,
) -> List[Dict[str, Any]]:
    """OpenAI multimodal user content with N inline frames + a text block.

    The ``schema_helpers["build_user_message"]`` helper only handles a
    single image, so for video clips we assemble the equivalent payload
    by base64-encoding each frame ourselves.
    """
    encode_image_b64 = schema_helpers["encode_image_b64"]
    parts: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames):
        try:
            b64 = encode_image_b64(fr, max_side=_VLM_IMAGE_MAX_SIDE)
        except Exception as exc:
            logger.debug("frame %d encode failed: %s", i, exc)
            continue
        parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })

    text_lines = [
        f"domain={domain}",
        f"task={task_id}",
        f"goal={goal}",
        f"step=0",
        f"num_frames={len(parts)}",
        "(frames are uniformly sampled from the video, ordered earliest -> latest)",
    ]
    if extra_context:
        text_lines.append("\nAdditional context:\n" + extra_context)
    parts.append({"type": "text", "text": "\n".join(text_lines)})
    return parts


def generate_schema_from_visual(
    *,
    image: Optional[Any],
    frames: Optional[Sequence[Any]],
    benchmark: str,
    modality: str,
    task_id: str,
    goal: str,
    question: str,
    valid_actions: Optional[List[str]],
    client: Any,
    routed_model: str,
    schema_helpers: Dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = _SCHEMA_MAX_TOKENS,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call gpt-5.5 (vision) -> ``<state>...</state>`` schema for one sample.

    For image benchmarks pass ``image=<PIL>`` and ``frames=None``.
    For video benchmarks pass ``frames=[<PIL>, ...]`` and ``image=None``;
    we send all frames as inline ``image_url`` parts so the model can
    reason temporally without a video tool registry.
    """
    domain = "image_qa" if modality == "image" else "video_qa"

    if client is None or schema_helpers is None:
        return {
            "schema": None, "raw": "", "source": "no_client",
            "error": None, "finish_reason": None, "recovery": "",
            "raw_full_len": 0,
        }

    if modality == "image" and image is None:
        return {
            "schema": None, "raw": "", "source": "no_image",
            "error": None, "finish_reason": None, "recovery": "",
            "raw_full_len": 0,
        }
    if modality == "video" and not frames:
        return {
            "schema": None, "raw": "", "source": "no_frames",
            "error": None, "finish_reason": None, "recovery": "",
            "raw_full_len": 0,
        }

    system = schema_helpers["build_system_prompt"](domain, max_entities=20)

    extra_parts: List[str] = [
        f"Benchmark: {benchmark} (modality={modality}).",
        f"Question:\n{question}",
    ]
    if valid_actions:
        extra_parts.append(
            "Allowed action set for downstream actor agent (reproduce these "
            "verbatim inside <actions>; do NOT invent new ones):\n"
            + "\n".join(f"  - {a}" for a in valid_actions[:25])
        )
    extra_context = "\n\n".join(extra_parts)

    if modality == "image":
        user_content = schema_helpers["build_user_message"](
            image,
            domain=domain,
            task_id=task_id,
            goal=goal,
            step=0,
            extra_context=extra_context,
        )
    else:
        user_content = _build_video_user_message(
            frames=list(frames or []),
            schema_helpers=schema_helpers,
            domain=domain,
            task_id=task_id,
            goal=goal,
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
        logger.warning("[schema-VLM] %s sample failed: %s", benchmark, exc)

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
                "[schema-VLM] %s salvaged via '%s' (finish_reason=%s, raw_len=%d)",
                benchmark, recovery, finish_reason, len(raw),
            )
        return {"schema": parsed, "source": "vlm", **base_meta}

    if raw and finish_reason == "length":
        logger.warning(
            "[schema-VLM] %s response truncated (finish_reason=length, raw_len=%d) "
            "— consider raising _SCHEMA_MAX_TOKENS",
            benchmark, len(raw),
        )

    return {"schema": None, "source": "vlm_no_schema", **base_meta}


# ---------------------------------------------------------------------------
# Stage 2 — schema-driven actor agent (gpt-5.5 -> answer)
# ---------------------------------------------------------------------------

_ACTOR_SYSTEM_PROMPT = (
    "You are an Actor Agent for the COS-PLAY visual-reasoning pipeline.\n"
    "On every step you receive (a) a structured ``<state>...</state>`` "
    "schema produced by a vision call on the input image / video frames, "
    "(b) the benchmark question, and (c) the set of valid actions / "
    "answers the environment will accept this step.\n\n"
    "Your job:\n"
    "1. Reason briefly (≤4 sentences) over the schema entities, "
    "evidence and answer hints — cite entity ids (e1, e2, ...) when they "
    "support the conclusion.\n"
    "2. Pick EXACTLY ONE answer from the valid-action list — copy the "
    "string verbatim (no renaming, no quoting, no reformatting).\n"
    "3. For multiple-choice video benchmarks the action MUST be a single "
    "letter from the allowed set (e.g. 'A', 'B', ...).  For free-form "
    "image benchmarks (visual_toolbench / tir_bench) the action must be "
    "a concise answer string (no explanation, no markdown).\n\n"
    "Always respond by calling the ``choose_answer`` function."
)


def _build_actor_tools(
    valid_actions: List[str],
    *,
    is_mcq: bool,
) -> list:
    """OpenAI function-calling tool definition for the actor."""
    answer_schema: Dict[str, Any] = {
        "type": "string",
        "description": (
            "EXACT verbatim string from the valid-action list."
            if is_mcq
            else "Concise answer string (e.g. a letter, number or short phrase)."
        ),
    }
    if is_mcq and valid_actions:
        answer_schema["enum"] = list(valid_actions)
    elif valid_actions:
        answer_schema["description"] += (
            f"  Allowed: {', '.join(valid_actions[:25])}"
        )

    return [
        {
            "type": "function",
            "function": {
                "name": "choose_answer",
                "description": "Choose the single answer for this benchmark sample.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Brief chain-of-thought (≤4 sentences) "
                                "grounded in the schema entities and evidence."
                            ),
                        },
                        "answer": answer_schema,
                    },
                    "required": ["answer"],
                },
            },
        }
    ]


def _canonicalize_action(raw: str, valid_actions: List[str]) -> Optional[str]:
    """Map ``raw`` to one of ``valid_actions`` (case-insensitive, prefix, substring)."""
    if not raw:
        return None
    cand = raw.strip().strip("`").strip('"').strip("'")
    if not cand:
        return None
    if cand in valid_actions:
        return cand
    lc = cand.lower()
    lower_map = {a.lower(): a for a in valid_actions}
    if lc in lower_map:
        return lower_map[lc]
    m = re.match(r"^\s*(\d+)\s*[\.\)\-:]?\s*$", cand)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]
    for a in valid_actions:
        if a.lower().startswith(lc) or lc.startswith(a.lower()):
            return a
    for a in valid_actions:
        if a.lower() in lc or lc in a.lower():
            return a
    return None


def select_action_from_schema(
    *,
    schema_text: Optional[str],
    question: str,
    options_block: Optional[str],
    valid_actions: List[str],
    is_mcq: bool,
    benchmark: str,
    client: Any,
    routed_model: str,
    temperature: float = 0.4,
    max_tokens: int = _ACTION_MAX_TOKENS,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Call gpt-5.5 with the schema -> ``(answer, reasoning, raw, error)``."""
    if client is None:
        return None, None, "", "no_client"
    if is_mcq and not valid_actions:
        return None, None, "", "no_valid_actions"

    schema_block = (
        schema_text.strip() if schema_text
        else "(no schema available — fall back to the auxiliary question text)"
    )

    user_parts = [
        f"Benchmark: {benchmark}",
        f"Question:",
        question.strip() if question else "(missing)",
    ]
    if options_block:
        user_parts.extend(["", options_block.strip()])
    user_parts.extend([
        "",
        "Structured visual-state schema (from gpt-5.5 vision call):",
        schema_block,
        "",
    ])
    if is_mcq:
        user_parts.append(
            f"Valid answers: {', '.join(valid_actions)}.  "
            "Pick EXACTLY one letter."
        )
    else:
        user_parts.append(
            "Free-form QA: answer concisely (a single phrase, number, "
            "or word).  Match the wording the question expects."
        )
    user_parts.extend([
        "",
        "Think step-by-step over the schema, then call the choose_answer function.",
    ])
    user_content = "\n".join(p for p in user_parts if p is not None)

    tools = _build_actor_tools(valid_actions, is_mcq=is_mcq)

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
            tool_choice={"type": "function", "function": {"name": "choose_answer"}},
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
            raw_answer = str(args.get("answer", "")).strip()
            reasoning = args.get("reasoning") or None
            if is_mcq:
                canonical = _canonicalize_action(raw_answer, valid_actions)
                if canonical:
                    return canonical, reasoning, raw or json.dumps(args), None
            else:
                if raw_answer:
                    return raw_answer, reasoning, raw or json.dumps(args), None

        # No tool call — try to extract from the raw message content.
        if is_mcq:
            canonical = _canonicalize_action(raw, valid_actions)
            if canonical:
                return canonical, None, raw, None
        elif raw.strip():
            return raw.strip(), None, raw, None

    except Exception as exc:
        err = repr(exc)
        logger.warning("[actor-LLM] %s sample failed: %s", benchmark, exc)

    return None, None, raw, err


# ---------------------------------------------------------------------------
# Per-benchmark sample iteration + frame extraction.
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSample:
    """Uniform shape we feed into the 2-stage pipeline."""

    benchmark: str
    modality: str  # "image" or "video"
    sample_id: str
    question: str
    gold_answer: Optional[str]
    valid_actions: List[str]
    is_mcq: bool
    options_block: Optional[str] = None  # rendered "Options:\nA. ...\nB. ..." for MCQ
    image: Any = None  # PIL.Image for image benchmarks
    frames: Optional[List[Any]] = None  # list of PIL.Image for video benchmarks
    video_meta: Optional[Dict[str, Any]] = None
    raw_sample: Optional[Dict[str, Any]] = None  # source-of-truth for debug

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "modality": self.modality,
            "sample_id": self.sample_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "valid_actions": list(self.valid_actions),
            "is_mcq": self.is_mcq,
            "options_block": self.options_block,
            "video_meta": self.video_meta,
            "raw_sample": self.raw_sample,
        }


def _iter_visual_toolbench_samples(num: int) -> Iterator[BenchmarkSample]:
    from visual_reasoning_wrapper.benchmarks.visual_toolbench import (
        iter_visual_toolbench_samples,
        load_visual_toolbench_image,
    )
    for s in iter_visual_toolbench_samples(limit=num, single_turn_only=True):
        try:
            pil = _resize_pil(load_visual_toolbench_image(s))
        except Exception as exc:
            logger.warning("VTB image decode failed for %s: %s", s.sample_id, exc)
            pil = None
        gold = (s.gold_answer or "") or None
        is_mcq, letters, options_block = _detect_inline_mcq(s.question or "", gold)
        yield BenchmarkSample(
            benchmark="visual_toolbench",
            modality="image",
            sample_id=str(s.sample_id),
            question=s.question or "",
            gold_answer=gold,
            valid_actions=letters,
            is_mcq=is_mcq,
            options_block=options_block,
            image=pil,
            raw_sample=s.to_dict(),
        )


def _iter_tir_bench_samples(num: int) -> Iterator[BenchmarkSample]:
    from visual_reasoning_wrapper.benchmarks.tir_bench import (
        iter_tir_bench_samples,
        load_tir_bench_image,
    )
    for s in iter_tir_bench_samples(limit=num):
        try:
            pil = _resize_pil(load_tir_bench_image(s))
        except Exception as exc:
            logger.warning("TIR image decode failed for %s: %s", s.sample_id, exc)
            pil = None
        gold = (s.answer or "") or None
        is_mcq, letters, options_block = _detect_inline_mcq(s.prompt or "", gold)
        yield BenchmarkSample(
            benchmark="tir_bench",
            modality="image",
            sample_id=str(s.sample_id),
            question=s.prompt or "",
            gold_answer=gold,
            valid_actions=letters,
            is_mcq=is_mcq,
            options_block=options_block,
            image=pil,
            raw_sample=s.to_dict(),
        )


def _format_options_block(options: Dict[str, str]) -> str:
    """Render an MCQ options table, sorted by letter."""
    if not options:
        return ""
    lines = ["Options:"]
    for letter in sorted(options.keys()):
        lines.append(f"{letter}. {options[letter]}")
    return "\n".join(lines)


def _iter_video_holmes_samples(num: int, *, num_frames: int) -> Iterator[BenchmarkSample]:
    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples,
        sample_video_frames,
    )
    yielded = 0
    for s in iter_video_holmes_samples(split="test", limit=num * 4):  # over-fetch in case some have no video
        if yielded >= num:
            break
        if not s.video_path or not Path(s.video_path).exists():
            logger.info("Video-Holmes %s.Q%s: missing video at %s, skipping",
                        s.video_id, s.question_id, s.video_path)
            continue
        try:
            frames, fps, vmeta = sample_video_frames(
                s.video_path, num_frames=num_frames, max_side=_VLM_IMAGE_MAX_SIDE,
            )
        except Exception as exc:
            logger.warning("Video-Holmes frame sample failed for %s: %s", s.video_id, exc)
            continue
        if not frames:
            continue
        # Restrict to letters that actually have an option.
        valid_letters = [L for L in _LETTERS_HOLMES if s.options.get(L) is not None]
        yield BenchmarkSample(
            benchmark="video_holmes",
            modality="video",
            sample_id=f"{s.video_id}.Q{s.question_id}",
            question=s.question or "",
            gold_answer=(s.answer or "") or None,
            valid_actions=valid_letters,
            is_mcq=True,
            options_block=_format_options_block(s.options),
            frames=frames,
            video_meta={
                "video_path": str(s.video_path),
                "question_type": s.question_type,
                **{k: v for k, v in (vmeta or {}).items() if k != "size"},
            },
            raw_sample=s.to_dict(),
        )
        yielded += 1


def _iter_siv_bench_samples(num: int, *, num_frames: int) -> Iterator[BenchmarkSample]:
    from visual_reasoning_wrapper.benchmarks.siv_bench import (
        iter_siv_bench_samples,
    )
    from visual_reasoning_wrapper.benchmarks.video_holmes import sample_video_frames
    yielded = 0
    for s in iter_siv_bench_samples(limit=num * 4):
        if yielded >= num:
            break
        if not s.video_path or not Path(s.video_path).exists():
            logger.info("SIV-Bench %s.Q%s: missing video, skipping",
                        s.video_id, s.question_id)
            continue
        try:
            frames, fps, vmeta = sample_video_frames(
                s.video_path, num_frames=num_frames, max_side=_VLM_IMAGE_MAX_SIDE,
            )
        except Exception as exc:
            logger.warning("SIV-Bench frame sample failed for %s: %s", s.video_id, exc)
            continue
        if not frames:
            continue
        valid_letters = [L for L in _LETTERS_SIV if s.options.get(L) is not None]
        yield BenchmarkSample(
            benchmark="siv_bench",
            modality="video",
            sample_id=f"{s.video_id}.Q{s.question_id}",
            question=s.question or "",
            gold_answer=(s.answer or "") or None,
            valid_actions=valid_letters,
            is_mcq=True,
            options_block=_format_options_block(s.options),
            frames=frames,
            video_meta={
                "video_path": str(s.video_path),
                "dimension": s.dimension,
                "subtask": s.subtask,
                "subtitle": s.subtitle,
                **{k: v for k, v in (vmeta or {}).items() if k != "size"},
            },
            raw_sample=s.to_dict(),
        )
        yielded += 1


BENCHMARK_ITERATORS: Dict[str, Callable[..., Iterator[BenchmarkSample]]] = {
    "visual_toolbench": lambda num, **kw: _iter_visual_toolbench_samples(num),
    "tir_bench":        lambda num, **kw: _iter_tir_bench_samples(num),
    "video_holmes":     _iter_video_holmes_samples,
    "siv_bench":        _iter_siv_bench_samples,
}

BENCHMARK_MODALITY: Dict[str, str] = {
    "visual_toolbench": "image",
    "tir_bench": "image",
    "video_holmes": "video",
    "siv_bench": "video",
}


# ---------------------------------------------------------------------------
# LLM-as-judge for free-form image QA  (VTB / TIR-Bench)
# ---------------------------------------------------------------------------
#
# Naïve substring matching breaks on long-form benchmarks: VTB ships
# multi-paragraph rubric-style golds and TIR-Bench occasionally asks for
# a numeric answer expressed differently from the gold (e.g. "1.14%" vs
# "1.15%").  Officially VTB grades by rubric and TIR-Bench by task-
# specific metric — both use a judge model under the hood.  We replicate
# that with a tiny gpt-5.5 judge call gated behind ``--judge``.
#
# The judge is *only* invoked for the free-form (non-MCQ) benchmarks;
# MCQ rows already grade exactly via letter equality and don't need it.
#
# Results are cached on disk under
# ``<output_dir>/<benchmark>/judge_cache/<sample_id>.json`` keyed by a
# hash of (gold, predicted, judge_model).  Re-runs are then free, so
# iterating on the prompt / cases doesn't re-spend on grading.
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402  (kept local to make the dependency obvious)


_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial grading judge for visual-reasoning benchmarks.\n"
    "You receive (a) the original question, (b) the gold reference answer "
    "(which may be long-form / rubric-style / multi-fact), and (c) a "
    "candidate prediction from another model.\n\n"
    "Decide whether the candidate prediction is SUBSTANTIVELY correct: it "
    "agrees with the gold on every fact the question demands, even if the "
    "wording, ordering, units, or level of detail differ.\n\n"
    "Grading rules:\n"
    "1. Numeric answers are correct when they round to / equal the gold "
    "value to a reasonable tolerance (≤2% relative for percentages and "
    "currency, ≤1 unit for integer counts).  Approximate match counts.\n"
    "2. Free-form answers are correct when every key fact the gold "
    "asserts is present (or directly implied) in the prediction, AND the "
    "prediction does not assert an incompatible claim.  Extra correct "
    "detail is fine; missing required detail is not.\n"
    "3. For diagnosis / classification tasks the named term in the "
    "prediction must agree with the named term (or unambiguous synonym) "
    "in the gold.\n"
    "4. If the gold is itself uninformative (empty / 'see rubric' / "
    "wholly subjective) and the prediction is a plausible answer, you "
    "may grade ``unscoreable`` — that is NOT counted as correct.\n\n"
    "Always respond by calling the ``grade_answer`` function."
)


def _judge_cache_key(gold: str, predicted: str, judge_model: str) -> str:
    """Deterministic cache key: hash(gold + predicted + judge_model)."""
    blob = json.dumps(
        {"gold": gold, "pred": predicted, "model": judge_model},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _judge_cache_path(cache_dir: Optional[Path], sample_id: str, key: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    safe = re.sub(r"[^\w.\-]+", "_", str(sample_id))[:80]
    return cache_dir / f"{safe}.{key}.json"


def llm_judge_correct(
    *,
    question: str,
    gold: str,
    predicted: str,
    benchmark: str,
    client: Any,
    routed_model: str,
    cache_dir: Optional[Path] = None,
    sample_id: str = "",
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> Dict[str, Any]:
    """Grade ``predicted`` against ``gold`` using a small gpt-5.5 judge.

    Returns ``{"correct": bool|None, "verdict": str, "reason": str,
    "cached": bool, "judge_model": str, "error": str|None}``.

    ``correct`` is ``None`` only when the call fails or the judge
    explicitly returns ``unscoreable``.
    """
    out_default = {
        "correct": None, "verdict": None, "reason": None,
        "cached": False, "judge_model": routed_model, "error": None,
    }
    if not gold or not predicted:
        out_default["error"] = "missing_gold_or_pred"
        return out_default

    key = _judge_cache_key(gold, predicted, routed_model)
    cache_path = _judge_cache_path(cache_dir, sample_id, key)
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cached"] = True
            return cached
        except Exception as exc:
            logger.debug("judge cache read failed at %s: %s", cache_path, exc)

    if client is None:
        out_default["error"] = "no_client"
        return out_default

    user_msg = (
        f"Benchmark: {benchmark}\n\n"
        f"Question:\n{question}\n\n"
        f"Gold reference answer:\n{gold}\n\n"
        f"Candidate prediction:\n{predicted}\n\n"
        "Apply the grading rules from the system prompt and call "
        "``grade_answer`` exactly once."
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "grade_answer",
                "description": "Grade a free-form prediction against a gold reference.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": ["correct", "incorrect", "unscoreable"],
                            "description": (
                                "correct = prediction agrees with all "
                                "required facts; incorrect = at least "
                                "one required fact is wrong / missing; "
                                "unscoreable = gold is not informative "
                                "enough to grade."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Two-sentence justification citing the "
                                "specific gold fact(s) the prediction "
                                "matches or fails to match."
                            ),
                        },
                    },
                    "required": ["verdict"],
                },
            },
        }
    ]

    try:
        resp = _chat_completion(
            client,
            model=routed_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "grade_answer"}},
        )
        msg = resp.choices[0].message
        verdict: Optional[str] = None
        reason: Optional[str] = None
        if getattr(msg, "tool_calls", None):
            tc = msg.tool_calls[0]
            raw_args = (
                getattr(tc, "arguments", None)
                or getattr(getattr(tc, "function", None), "arguments", None)
                or "{}"
            )
            args_d = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            verdict = (args_d.get("verdict") or "").strip().lower() or None
            reason = args_d.get("reason") or None
        if verdict not in ("correct", "incorrect", "unscoreable"):
            return {
                "correct": None, "verdict": None,
                "reason": (msg.content or "")[:300] or "no tool_call returned",
                "cached": False, "judge_model": routed_model,
                "error": "unparsed_judge_response",
            }
        correct = True if verdict == "correct" else False if verdict == "incorrect" else None
        out = {
            "correct": correct, "verdict": verdict, "reason": reason,
            "cached": False, "judge_model": routed_model, "error": None,
        }
    except Exception as exc:
        logger.warning("[judge] %s sample %s failed: %s", benchmark, sample_id, exc)
        out = {
            "correct": None, "verdict": None, "reason": None,
            "cached": False, "judge_model": routed_model,
            "error": repr(exc),
        }

    if cache_path is not None and out.get("error") is None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(out, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("judge cache write failed at %s: %s", cache_path, exc)
    return out


# Benchmarks where ``--judge`` actually runs (the MCQ benchmarks grade
# exactly via letter equality so a judge call adds zero signal).
_JUDGE_BENCHMARKS = {"visual_toolbench", "tir_bench"}


# ---------------------------------------------------------------------------
# Per-sample driver + per-benchmark batch driver.
# ---------------------------------------------------------------------------

def _is_correct(predicted: Optional[str], gold: Optional[str], *, is_mcq: bool) -> Optional[bool]:
    """String-match diagnostic correctness (None if either side is missing)."""
    if predicted is None or gold is None:
        return None
    p = str(predicted).strip()
    g = str(gold).strip()
    if not p or not g:
        return None
    if is_mcq:
        return p.upper() == g.upper()
    pl, gl = p.lower(), g.lower()
    return pl == gl or gl in pl or pl in gl


def _run_one_sample(
    sample: BenchmarkSample,
    *,
    args: argparse.Namespace,
    client: Any,
    routed_model: str,
    schema_helpers: Optional[Dict[str, Any]],
    frames_dir: Optional[Path],
    judge_cache_dir: Optional[Path] = None,
    judge_routed_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the (vision -> schema -> action) pipeline on one sample."""
    task_id = f"{sample.benchmark}.{sample.sample_id}"
    goal = (sample.question or "").strip().split("\n")[0]

    # Persist frames sent to the VLM (debug).
    saved_frames: List[str] = []
    if frames_dir is not None and args.save_frames:
        if sample.modality == "image" and sample.image is not None:
            p = _save_pil(sample.image, frames_dir / "frame_00.png")
            if p:
                saved_frames.append(p)
        elif sample.modality == "video" and sample.frames:
            for i, fr in enumerate(sample.frames):
                p = _save_pil(fr, frames_dir / f"frame_{i:02d}.png")
                if p:
                    saved_frames.append(p)

    # Stage 1 — vision -> schema (skipped under --no_vision).
    if not args.no_vision and schema_helpers is not None and client is not None:
        budget = (
            _SCHEMA_MAX_TOKENS_REASONING
            if _is_reasoning_model(routed_model)
            else _SCHEMA_MAX_TOKENS
        )
        schema_meta = generate_schema_from_visual(
            image=sample.image,
            frames=sample.frames,
            benchmark=sample.benchmark,
            modality=sample.modality,
            task_id=task_id,
            goal=goal,
            question=sample.question,
            valid_actions=sample.valid_actions,
            client=client,
            routed_model=routed_model,
            schema_helpers=schema_helpers,
            temperature=args.temperature_schema,
            max_tokens=budget,
            reasoning_effort=getattr(args, "reasoning_effort", None),
        )
    else:
        schema_meta = {
            "schema": None,
            "raw": "",
            "raw_full_len": 0,
            "source": "skipped",
            "error": None,
            "finish_reason": None,
            "recovery": "",
        }

    schema_text = schema_meta.get("schema")

    # Stage 2 — schema -> action.
    answer, reasoning, action_raw, action_err = select_action_from_schema(
        schema_text=schema_text,
        question=sample.question,
        options_block=sample.options_block,
        valid_actions=sample.valid_actions,
        is_mcq=sample.is_mcq,
        benchmark=sample.benchmark,
        client=client,
        routed_model=routed_model,
        temperature=args.temperature_action,
        reasoning_effort=getattr(args, "reasoning_effort", None),
    )
    correct_strmatch = _is_correct(answer, sample.gold_answer, is_mcq=sample.is_mcq)

    # Stage 3 (optional) — LLM-as-judge for free-form benchmarks.
    judge_meta: Optional[Dict[str, Any]] = None
    correct = correct_strmatch
    if (
        getattr(args, "judge", False)
        and not sample.is_mcq
        and sample.benchmark in _JUDGE_BENCHMARKS
        and answer is not None
        and sample.gold_answer is not None
    ):
        judge_meta = llm_judge_correct(
            question=sample.question,
            gold=sample.gold_answer,
            predicted=answer,
            benchmark=sample.benchmark,
            client=client,
            routed_model=(judge_routed_model or routed_model),
            cache_dir=judge_cache_dir,
            sample_id=sample.sample_id,
        )
        if judge_meta.get("correct") is not None:
            correct = bool(judge_meta["correct"])

    record: Dict[str, Any] = {
        "benchmark": sample.benchmark,
        "modality": sample.modality,
        "sample_id": sample.sample_id,
        "task_id": task_id,
        "question": sample.question,
        "gold_answer": sample.gold_answer,
        "is_mcq": sample.is_mcq,
        "valid_actions": list(sample.valid_actions),
        "options_block": sample.options_block,
        "answer": answer,
        "answer_reasoning": reasoning,
        "answer_raw": (action_raw or "")[:4000] if action_raw else None,
        "answer_error": action_err,
        "correct": correct,
        "correct_strmatch": correct_strmatch,
        "judge": judge_meta,
        "schema": schema_text,
        "schema_source": schema_meta.get("source"),
        "schema_finish_reason": schema_meta.get("finish_reason"),
        "schema_recovery": schema_meta.get("recovery") or None,
        "schema_error": schema_meta.get("error"),
        "schema_raw_excerpt": (schema_meta.get("raw") or "")[:4000] or None,
        "schema_raw_full_len": schema_meta.get("raw_full_len"),
        "video_meta": sample.video_meta,
        "frames_saved": saved_frames or None,
        "raw_sample": sample.raw_sample,
        "model": args.model,
        "model_routed": routed_model,
    }
    return record


def run_benchmark(
    benchmark: str,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    client: Any,
    routed_model: str,
    schema_helpers: Optional[Dict[str, Any]],
    judge_routed_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the configured number of test cases for one benchmark."""
    bench_dir = output_dir / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = bench_dir / "samples.jsonl"
    summary_path = bench_dir / "summary.json"
    modality = BENCHMARK_MODALITY[benchmark]
    judge_cache_dir: Optional[Path] = None
    if getattr(args, "judge", False) and benchmark in _JUDGE_BENCHMARKS:
        judge_cache_dir = bench_dir / "judge_cache"
        judge_cache_dir.mkdir(parents=True, exist_ok=True)

    iter_factory = BENCHMARK_ITERATORS[benchmark]
    samples_iter: Iterator[BenchmarkSample]

    sample_id_filter: Optional[set] = _load_sample_id_filter(
        getattr(args, "sample_ids_dir", None), benchmark
    )
    if sample_id_filter:
        # Make both the inner wrapper budget and the outer per-benchmark
        # cap big enough that we never short-circuit before the filter
        # has matched every requested id. 10_000 > the largest source
        # pool (SIV-Bench = 8,728) so this exhausts the dataset.
        iterator_budget = 10_000
        if args.num_test_cases < len(sample_id_filter):
            args.num_test_cases = len(sample_id_filter)
    else:
        iterator_budget = args.num_test_cases

    try:
        samples_iter = iter_factory(iterator_budget, num_frames=args.num_frames)
        if sample_id_filter:
            samples_iter = (
                s for s in samples_iter if str(s.sample_id) in sample_id_filter
            )
    except Exception as exc:
        msg = f"sample iterator setup failed: {exc!r}"
        logger.error("[%s] %s", benchmark, msg)
        if args.verbose:
            traceback.print_exc()
        summary = {
            "benchmark": benchmark,
            "modality": modality,
            "fatal_error": msg,
            "samples_attempted": 0,
            "samples_completed": 0,
        }
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
        return summary

    records: List[Dict[str, Any]] = []
    schema_ok = 0
    answer_ok = 0
    correct_ok = 0
    correct_total = 0
    errors = 0
    started = time.time()

    # ----- Per-sample worker (used by both serial and threadpool paths) -----
    def _process_sample(idx: int, sample: BenchmarkSample) -> Dict[str, Any]:
        """Run one sample end-to-end and return the record dict.

        ``idx`` is the *input order* index — used for the
        ``sample_NNN.json`` filename so artifacts stay deterministic
        even when results arrive out-of-order from a thread pool.
        """
        out_path = bench_dir / f"sample_{idx:03d}.json"
        # Idempotent skip: if a successful per-sample JSON already exists for
        # this sample_id, reuse it instead of paying the LLM cost again. This
        # makes external relaunches into the same dir cheap and lets a later
        # FIFO-dispatched leg short-circuit when the work was done out-of-band.
        if out_path.exists():
            try:
                with out_path.open("r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                same_id = str(cached.get("sample_id")) == str(sample.sample_id)
                successful = "error" not in cached and cached.get("answer") is not None
                if same_id and successful:
                    cached["sample_index"] = idx
                    cached["resumed"] = True
                    return cached
            except Exception:
                pass  # fall through and re-run
        frames_dir = (
            bench_dir / "frames" / f"sample_{idx:03d}" if args.save_frames else None
        )
        t0 = time.time()
        try:
            record = _run_one_sample(
                sample,
                args=args,
                client=client,
                routed_model=routed_model,
                judge_routed_model=judge_routed_model,
                schema_helpers=schema_helpers,
                frames_dir=frames_dir,
                judge_cache_dir=judge_cache_dir,
            )
        except Exception as exc:
            logger.error("[%s] sample %s pipeline failed: %s",
                         benchmark, sample.sample_id, exc)
            if args.verbose:
                traceback.print_exc()
            record = {
                "benchmark": benchmark,
                "modality": modality,
                "sample_id": sample.sample_id,
                "error": repr(exc),
                "raw_sample": sample.raw_sample,
            }
        record["sample_index"] = idx
        record["elapsed_seconds"] = round(time.time() - t0, 3)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False, default=str)
        return record

    # ----- Aggregator: shared by both paths -----
    write_lock = threading.Lock()  # serializes JSONL appends + counter updates

    def _aggregate(record: Dict[str, Any], jfh) -> None:
        nonlocal schema_ok, answer_ok, correct_ok, correct_total, errors
        with write_lock:
            if record.get("schema"):
                schema_ok += 1
            if record.get("answer") is not None:
                answer_ok += 1
            corr = record.get("correct")
            if corr is True:
                correct_ok += 1
                correct_total += 1
            elif corr is False:
                correct_total += 1
            if "error" in record:
                errors += 1
            records.append(record)
            jfh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            jfh.flush()
            if args.verbose:
                ans = record.get("answer")
                src = record.get("schema_source")
                gold = record.get("gold_answer")
                tag = (
                    "OK" if record.get("correct") is True
                    else "NO" if record.get("correct") is False
                    else "??"
                )
                judge_meta = record.get("judge") or {}
                judge_tag = ""
                if judge_meta:
                    v = judge_meta.get("verdict")
                    cached = judge_meta.get("cached")
                    if v:
                        judge_tag = f" judge={v}{'(cached)' if cached else ''}"
                print(
                    f"  [{benchmark}] sample {record.get('sample_index')+1}: "
                    f"{record.get('sample_id')!r:<24} "
                    f"-> answer={ans!r:<20} gold={gold!r:<14} "
                    f"correct={tag} schema={src}{judge_tag} "
                    f"({record['elapsed_seconds']:.1f}s)"
                )

    num_workers = max(1, int(getattr(args, "num_workers", 1) or 1))

    with jsonl_path.open("w", encoding="utf-8") as jfh:
        if num_workers == 1:
            # ----- Serial path (backward-compatible) -----
            idx = 0
            while True:
                try:
                    sample = next(samples_iter)
                except StopIteration:
                    break
                except Exception as exc:
                    errors += 1
                    logger.error("[%s] sample iteration failed: %s", benchmark, exc)
                    if args.verbose:
                        traceback.print_exc()
                    break
                if not args.verbose:
                    print(f"  [{benchmark}] sample {idx + 1}/{args.num_test_cases}: "
                          f"{sample.sample_id}")
                record = _process_sample(idx, sample)
                _aggregate(record, jfh)
                idx += 1
                if idx >= args.num_test_cases:
                    break
        else:
            # ----- Threadpool path -----
            # Materialize the (capped) sample list first.  Iterators in
            # ``BENCHMARK_ITERATORS`` are lazy and may be backed by HF
            # ``streaming=True`` datasets, so this is the only place
            # we eagerly pull samples.  Memory footprint stays bounded
            # by ``num_test_cases`` (≤ 1,000 in the lean plan).
            #
            # Iteration is wrapped because a streaming HF dataset can
            # raise mid-pull (corrupt video, network blip, S3 throttle).
            # Mirror the serial path: count the failure as an error and
            # stop pulling — already-pulled samples are still dispatched.
            materialized: List[Tuple[int, BenchmarkSample]] = []
            iterator_failed = False
            while not iterator_failed and len(materialized) < args.num_test_cases:
                try:
                    s = next(samples_iter)
                except StopIteration:
                    break
                except Exception as exc:
                    errors += 1
                    iterator_failed = True
                    logger.error(
                        "[%s] sample iteration failed after %d samples: %s",
                        benchmark, len(materialized), exc,
                    )
                    if args.verbose:
                        traceback.print_exc()
                    break
                materialized.append((len(materialized), s))
            print(f"  [{benchmark}] dispatching {len(materialized)} samples "
                  f"to {num_workers} workers")
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [
                    pool.submit(_process_sample, idx, s) for idx, s in materialized
                ]
                completed = 0
                for fut in as_completed(futures):
                    completed += 1
                    try:
                        record = fut.result()
                    except Exception as exc:
                        with write_lock:
                            errors += 1
                        logger.error("[%s] worker raised: %s", benchmark, exc)
                        continue
                    _aggregate(record, jfh)
                    if not args.verbose and completed % max(1, len(materialized) // 20) == 0:
                        print(f"  [{benchmark}] {completed}/{len(materialized)} done")

    elapsed = time.time() - started
    judge_used = sum(1 for r in records if r.get("judge"))
    judge_cached = sum(
        1 for r in records
        if (r.get("judge") or {}).get("cached")
    )
    judge_unscoreable = sum(
        1 for r in records
        if (r.get("judge") or {}).get("verdict") == "unscoreable"
    )
    summary = {
        "benchmark": benchmark,
        "modality": modality,
        "samples_attempted": len(records),
        "samples_completed": len([r for r in records if "error" not in r]),
        "schema_ok": schema_ok,
        "answer_ok": answer_ok,
        "correct_ok": correct_ok,
        "correct_total_with_gold": correct_total,
        "accuracy": (correct_ok / correct_total) if correct_total else None,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 2),
        "model": args.model,
        "model_routed": routed_model,
        "use_vision": not args.no_vision,
        "num_test_cases": args.num_test_cases,
        "num_frames": args.num_frames if modality == "video" else None,
        "judge_enabled": bool(getattr(args, "judge", False) and benchmark in _JUDGE_BENCHMARKS),
        "judge_calls": judge_used,
        "judge_cached": judge_cached,
        "judge_unscoreable": judge_unscoreable,
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_benchmarks(requested: List[str]) -> Tuple[List[str], List[str]]:
    out: List[str] = []
    skipped: List[str] = []
    seen = set()
    for name in requested:
        key = name.strip().lower()
        if key in BENCHMARK_ITERATORS:
            if key not in seen:
                out.append(key)
                seen.add(key)
        else:
            skipped.append(name)
    return out, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cold-start actor-agent rollouts on the four visual-reasoning "
            "benchmarks (gpt-5.5 vision -> <state> schema -> actor agent)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="+", default=list(DEFAULT_BENCHMARKS),
        help=(
            "Benchmarks to run (default: all four). "
            f"Allowed: {', '.join(DEFAULT_BENCHMARKS)}"
        ),
    )
    parser.add_argument(
        "--num_test_cases", "--num-test-cases", "-n",
        type=int, default=DEFAULT_NUM_TEST_CASES,
        help=f"Test cases per benchmark (default: {DEFAULT_NUM_TEST_CASES}).",
    )
    parser.add_argument(
        "--sample_ids_dir", "--sample-ids-dir", type=str, default=None,
        help=(
            "Directory of sample-id manifest files to filter the per-benchmark "
            "iterators. Looks for <benchmark>_*.txt or <benchmark>.txt. "
            "Generated by cold_start/task_samples/build_visual_reasoning_diverse_1000.py. "
            "Implies --num_test_cases is bumped to the manifest size."
        ),
    )
    parser.add_argument(
        "--num_frames", "--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
        help=f"Frames per video clip (default: {DEFAULT_NUM_FRAMES}).",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Backbone model for vision + actor (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature_schema", type=float, default=0.2,
        help="Sampling temperature for the visual schema call (default: 0.2).",
    )
    parser.add_argument(
        "--temperature_action", type=float, default=0.4,
        help="Sampling temperature for the actor call (default: 0.4).",
    )
    parser.add_argument(
        "--reasoning_effort", "--reasoning-effort",
        type=str, default=None,
        choices=list(_VALID_REASONING_EFFORTS),
        help=(
            "OpenAI reasoning_effort knob for gpt-5.x / o1 / o3 / o4. "
            "One of {minimal, low, medium, high}. Default: unset (= "
            "OpenAI default 'medium'). For visual MCQ benchmarks, "
            "'medium' is the safer default since hidden CoT helps "
            "multi-hop social-causal inference and tool-use composition. "
            "Drop to 'minimal' only if a paired smoke test confirms no "
            "accuracy regression."
        ),
    )
    parser.add_argument(
        "--num_workers", "--num-workers", "-w",
        type=int, default=1,
        help=(
            "Number of concurrent samples to dispatch to the OpenAI API. "
            "Each sample is a pure-API workflow with no shared local "
            "state, so a ThreadPoolExecutor is safe and embarrassingly "
            "parallel. Default 1 (serial). Recommended for the lean "
            "plan: 16-32 for tier-4 OpenAI accounts (~10 k RPM), 32-64 "
            "for tier-5 (~30 k RPM). Set to 1 to debug or when the "
            "judge cache is being warmed."
        ),
    )
    parser.add_argument(
        "--no_vision", action="store_true",
        help="Skip the vision call; the actor sees only the question text.",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help=(
            "Enable LLM-as-judge for free-form scoring on "
            "VisualToolBench / TIR-Bench (substring match is too "
            "strict for these benchmarks).  Verdicts cached on disk "
            "under <benchmark>/judge_cache/, so re-runs are free."
        ),
    )
    parser.add_argument(
        "--judge_model", "--judge-model", type=str, default=None,
        help=(
            "Model used for --judge calls.  Defaults to --model.  "
            "A small reasoning model (e.g. gpt-5.4-mini, gpt-5-mini) "
            "is usually sufficient and cuts judge cost ~70-90%%."
        ),
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="Persist images / video frames sent to the VLM under "
             "<benchmark>/frames/sample_NNN/.",
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
        help="Output directory (default: <codebase_root>/Cold-start-out-visual-reasoning/<run_id>).",
    )
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Override auto-generated YYYY-MM-DD_HH-MM-SS run id.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-sample details (answer, gold, schema source).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    benchmarks, skipped = _resolve_benchmarks(args.benchmarks)
    if not benchmarks:
        print("[ERROR] No valid benchmarks selected. Available: "
              + ", ".join(BENCHMARK_ITERATORS.keys()))
        return 2

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = CODEBASE_ROOT / "Cold-start-out-visual-reasoning" / run_id
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
        model=args.model, api_key=args.api_key, base_url=args.base_url,
    )
    if client is None:
        print("[WARNING] No OpenAI/OpenRouter client could be built — pipeline will run with empty results.")

    # Resolve the judge routing target.  When ``--judge_model`` is
    # unset we route the judge through ``args.model``; otherwise we
    # pass the override through ``effective_openai_model`` so that
    # OpenRouter routing (``openai/<slug>``) is applied consistently.
    judge_model_cfg = (args.judge_model or args.model)
    _, judge_routed_model = _build_client_and_route(
        model=judge_model_cfg, api_key=args.api_key, base_url=args.base_url,
    )

    schema_helpers = _import_schema_helpers()

    print("=" * 78)
    print("  Cold-Start Actor — Visual Reasoning Benchmarks (gpt-5.5)")
    print("=" * 78)
    if _API_KEYS_FILE_USED is not None:
        print(f"  API keys file:      {_API_KEYS_FILE_USED}")
    print(f"  Run id:             {run_id}")
    print(f"  Benchmarks:         {', '.join(benchmarks)}")
    if skipped:
        print(f"  Skipped (unknown):  {', '.join(skipped)}")
    print(f"  Test cases:         {args.num_test_cases} per benchmark")
    print(f"  Frames per video:   {args.num_frames}")
    print(f"  Model (configured): {args.model}")
    print(f"  Model (routed):     {routed_model}")
    print(f"  Vision schema:      {'OFF (--no_vision)' if args.no_vision else 'ON'}")
    if args.judge:
        if judge_routed_model != routed_model:
            print(f"  LLM-as-judge:       ON  (model: {judge_model_cfg} -> {judge_routed_model})")
        else:
            print(f"  LLM-as-judge:       ON  (model: same as actor)")
    else:
        print(f"  LLM-as-judge:       OFF")
    print(f"  Save frames:        {args.save_frames}")
    print(f"  Output:             {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    per_benchmark: List[Dict[str, Any]] = []
    for bench in benchmarks:
        print(f"\n{'━' * 78}")
        print(f"  BENCHMARK: {bench} ({BENCHMARK_MODALITY[bench]})")
        print(f"{'━' * 78}")
        summary = run_benchmark(
            bench,
            args=args,
            output_dir=output_dir,
            client=client,
            routed_model=routed_model,
            judge_routed_model=judge_routed_model,
            schema_helpers=schema_helpers,
        )
        per_benchmark.append(summary)
        if "fatal_error" in summary:
            print(f"  [FATAL] {bench}: {summary['fatal_error']}")
            continue
        print(
            f"  [DONE]  {bench}: "
            f"completed={summary['samples_completed']}/{summary['samples_attempted']} "
            f"schema_ok={summary['schema_ok']} answer_ok={summary['answer_ok']} "
            f"correct={summary['correct_ok']}/{summary['correct_total_with_gold']} "
            f"({summary['elapsed_seconds']}s)"
        )

    overall_elapsed = time.time() - overall_t0

    # Update / refresh latest symlink (best-effort).
    try:
        latest = output_dir.parent / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(output_dir.name)
    except Exception:
        pass

    master = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_routed": routed_model,
        "use_vision": not args.no_vision,
        "judge_enabled": bool(args.judge),
        "judge_model": (args.judge_model or args.model) if args.judge else None,
        "judge_model_routed": judge_routed_model if args.judge else None,
        "num_test_cases": args.num_test_cases,
        "num_frames": args.num_frames,
        "benchmarks": list(benchmarks),
        "skipped_benchmarks": skipped,
        "elapsed_seconds": round(overall_elapsed, 2),
        "per_benchmark": per_benchmark,
    }
    master_path = output_dir / "batch_summary.json"
    with master_path.open("w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 78)
    print("  VISUAL-REASONING ACTOR — BATCH COMPLETE")
    print("=" * 78)
    total_eps = sum(b.get("samples_completed", 0) for b in per_benchmark)
    correct_eps = sum(b.get("correct_ok") or 0 for b in per_benchmark)
    correct_with_gold = sum(b.get("correct_total_with_gold") or 0 for b in per_benchmark)
    print(f"  Benchmarks processed: {len(per_benchmark)}")
    print(f"  Samples completed:    {total_eps}")
    if correct_with_gold:
        acc = correct_eps / correct_with_gold
        print(f"  Aggregate accuracy:   {correct_eps}/{correct_with_gold} = {acc:.2%}")
    print(f"  Elapsed:              {overall_elapsed:.1f}s")
    print(f"  Output:               {output_dir}")
    print(f"  Master summary:       {master_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
