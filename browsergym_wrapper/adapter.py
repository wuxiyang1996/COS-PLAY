"""BrowserGym adapter: screenshot → vision LLM → structured schema.

Sends the browser screenshot to a vision LLM (GPT-4o by default) and
receives the canonical <state>…</state> schema.  The AXTree / DOM text
is optionally attached as grounding context so GPT can emit correct
element bids, but the **screenshot is the primary input**.

Two modes:
  1. ``generate_label``  — offline batch label generation for training data.
  2. ``browser_obs_to_schema`` — convenience function that unpacks a
     BrowserGym obs dict and calls generate_label.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import openai
from PIL import Image

from vlm_wrapper.schema import (
    build_system_prompt,
    build_user_message,
    parse_schema_output,
    semantic_validate,
    validate_schema,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("VLM_LABEL_MODEL", "gpt-4o")
_DEFAULT_MAX_TOKENS = int(os.environ.get("VLM_LABEL_MAX_TOKENS", "1200"))
_DEFAULT_TEMPERATURE = float(os.environ.get("VLM_LABEL_TEMPERATURE", "0.2"))


# ======================================================================
# Core function: image → GPT-4o → schema
# ======================================================================

def generate_label(
    image: Image.Image | np.ndarray,
    *,
    goal: str = "",
    task_id: str = "",
    step: int = 0,
    url: str = "",
    axtree_text: str = "",
    last_action: str = "",
    last_action_error: str = "",
    max_entities: int = 25,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    """Send a browser screenshot to a vision LLM and return the structured schema.

    Parameters
    ----------
    image : PIL.Image or np.ndarray
        The browser screenshot (visual input).
    goal : str
        Task instruction (from ``obs["goal"]``).
    task_id : str
        Task identifier (e.g. ``"webarena.shopping.143"``).
    step : int
        Current step number.
    url : str
        Current page URL — passed as text context.
    axtree_text : str
        Flattened AXTree string — passed as grounding context so GPT can
        use correct bids and element roles.  NOT the primary input.
    last_action / last_action_error : str
        Previous action and any error — passed as context.
    max_entities : int
        Entity cap for the schema output.
    model : str
        Vision model to call (default: ``$VLM_LABEL_MODEL`` or ``gpt-4o``).
    api_key / base_url : str
        Override the OpenAI client config.
    temperature / max_tokens : float / int
        Generation parameters.
    retries : int
        Number of retries on parse failure.

    Returns
    -------
    dict with keys:
        ``"schema"``   – the ``<state>…</state>`` string (or None on failure)
        ``"raw"``      – raw LLM output before parsing
        ``"warnings"`` – list of validation warnings
        ``"model"``    – model used
    """
    model = model or _DEFAULT_MODEL
    temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
    max_tokens = max_tokens or _DEFAULT_MAX_TOKENS

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    system = build_system_prompt("browser", max_entities=max_entities)

    extra_ctx_parts: list[str] = []
    if url:
        extra_ctx_parts.append(f"URL: {url}")
    if last_action:
        extra_ctx_parts.append(f"Last action: {last_action}")
    if last_action_error:
        extra_ctx_parts.append(f"Last action error: {last_action_error}")
    if axtree_text:
        trimmed = axtree_text[:3000]
        extra_ctx_parts.append(
            f"AXTree (for element bid grounding, truncated):\n{trimmed}"
        )
    extra_context = "\n".join(extra_ctx_parts)

    user_content = build_user_message(
        image,
        domain="browser",
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
    warnings: list[str] = []
    validation: dict[str, Any] | None = None

    for attempt in range(1, retries + 2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content or ""
            schema = parse_schema_output(raw)
            if schema:
                warnings = validate_schema(schema)
                vres = semantic_validate(
                    schema,
                    domain="browser",
                    image_size=image.size if hasattr(image, "size") else None,
                )
                validation = vres.as_dict()
                warnings = warnings + vres.warnings + vres.errors
                break
            else:
                logger.warning("Attempt %d: no <state> block in GPT output", attempt)
        except Exception as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)
            raw = f"Error: {exc}"

    return {
        "schema": schema,
        "raw": raw,
        "warnings": warnings,
        "validation": validation,
        "model": model,
    }


# ======================================================================
# Convenience: unpack a BrowserGym obs dict
# ======================================================================

def browser_obs_to_schema(
    obs: dict[str, Any],
    *,
    step: int = 0,
    task_id: str = "",
    axtree_text: str = "",
    max_entities: int = 25,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper that unpacks a BrowserGym obs dict.

    Parameters
    ----------
    obs : dict
        The observation dict from ``BrowserEnv._get_obs()``.
    axtree_text : str
        Pre-flattened AXTree text.  If empty, the AXTree will NOT be
        included (caller should flatten it themselves using
        ``browsergym.utils.obs.flatten_axtree_to_str`` if desired).
    step, task_id, max_entities, model, api_key, base_url :
        Forwarded to ``generate_label``.

    Returns
    -------
    dict — same as ``generate_label``.
    """
    screenshot = obs.get("screenshot")
    if screenshot is None:
        return {"schema": None, "raw": "", "warnings": ["no screenshot"], "model": model or _DEFAULT_MODEL}

    if isinstance(screenshot, np.ndarray):
        image = Image.fromarray(screenshot)
    elif isinstance(screenshot, Image.Image):
        image = screenshot
    else:
        return {"schema": None, "raw": "", "warnings": ["unknown screenshot type"], "model": model or _DEFAULT_MODEL}

    goal = obs.get("goal", "")
    if not goal:
        goal_obj = obs.get("goal_object", ())
        texts = [m.get("text", "") for m in goal_obj if m.get("type") == "text"]
        goal = " ".join(texts)

    return generate_label(
        image,
        goal=goal,
        task_id=task_id,
        step=step,
        url=obs.get("url", ""),
        axtree_text=axtree_text,
        last_action=obs.get("last_action", ""),
        last_action_error=obs.get("last_action_error", ""),
        max_entities=max_entities,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
