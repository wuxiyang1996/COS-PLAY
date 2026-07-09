"""OSWorld adapter: desktop screenshot → vision LLM → structured schema.

Head 2 vision adapter for the ``desktop`` domain (OSWorld and any
captured OS-level screenshot). Mirrors the shape of
:mod:`browsergym_wrapper.adapter` so the calling code can treat all
five domains (gymv / browser / desktop / image_qa / video_qa) uniformly.

Two public entry points:
  1. ``generate_label`` — pass a raw screenshot plus optional grounding
     context (a11y tree, terminal output, task instruction) and get the
     canonical ``<state>…</state>`` schema back.
  2. ``osworld_obs_to_schema`` — convenience wrapper that unpacks an
     observation dict from ``OSWorldGymWrapper``.

The adapter uses ``build_adaptive_system_prompt(domain="desktop", …)``
with the full interactive section list so the schema header carries
``domain=desktop`` and the rules mention pixel coordinates +
pyautogui-style actions. Because OSWorld's dominant control surface is
``pyautogui.*``, the ``<actions>`` section is tuned accordingly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import openai
from PIL import Image

from vlm_wrapper.few_shot_library import get_few_shot_examples
from vlm_wrapper.schema import (
    build_adaptive_system_prompt,
    build_user_message,
    parse_schema_output,
    semantic_validate,
    validate_schema,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("VLM_LABEL_MODEL", "gpt-4o")
_DEFAULT_MAX_TOKENS = int(os.environ.get("VLM_LABEL_MAX_TOKENS", "1200"))
_DEFAULT_TEMPERATURE = float(os.environ.get("VLM_LABEL_TEMPERATURE", "0.2"))

_DESKTOP_SECTIONS = [
    "entities", "attributes", "relations",
    "state_flags", "targets", "uncertainty", "evidence", "actions",
]

_DESKTOP_ACTION_HINT = (
    "Valid OSWorld actions are pyautogui commands "
    "(pyautogui.click(x, y), pyautogui.doubleClick(x, y), "
    "pyautogui.typewrite('text'), pyautogui.hotkey('ctrl', 's'), "
    "pyautogui.scroll(-3), pyautogui.press('enter')) plus the special "
    "tokens DONE / FAIL / WAIT. Use absolute pixel coordinates for "
    "click/move targets."
)


# ======================================================================
# Core function: screenshot → vision LLM → schema
# ======================================================================

def generate_label(
    image: Image.Image | np.ndarray,
    *,
    instruction: str = "",
    goal: str = "",
    task_id: str = "",
    step: int = 0,
    a11y_tree_xml: str = "",
    terminal_output: str = "",
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
    """Send a desktop screenshot to a vision LLM and return the structured schema.

    Parameters
    ----------
    image : PIL.Image or np.ndarray
        The desktop / OS-level screenshot.
    instruction : str
        Task description from OSWorld (``obs["instruction"]``).
    goal : str
        Shorter goal if available; falls back to ``instruction``.
    task_id : str
        Identifier such as ``"osworld.install-spotify"``.
    step : int
        Current step number.
    a11y_tree_xml : str
        The OS accessibility tree (XML). Truncated before being attached
        as text context so the VLM can reference role/name pairs but is
        not overwhelmed.
    terminal_output : str
        Recent terminal output when relevant. Trimmed to the last lines.
    last_action / last_action_error : str
        Previous pyautogui command and any error raised executing it.
    max_entities : int
        Entity cap for the schema output.
    model, api_key, base_url, temperature, max_tokens : see
        :func:`browsergym_wrapper.adapter.generate_label`.
    retries : int
        Number of retries on parse failure.

    Returns
    -------
    dict with keys ``schema``, ``raw``, ``warnings``, ``model``.
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

    # 1-shot ICL wiring (T2.13', 2026-05-03): inject the curated desktop
    # example so base-VLM and base-Qwen3.5 inference both see the canonical
    # ``<state>`` ontology — closes the gap to the schema_gen-LoRA-tuned
    # variant without per-domain SFT.  Off via ``VLM_FEW_SHOT_N=0``.
    try:
        _few_shot_n = int(os.environ.get("VLM_FEW_SHOT_N", "1"))
    except ValueError:
        _few_shot_n = 1
    _examples = get_few_shot_examples(
        "desktop", n=_few_shot_n, task_id=task_id,
    ) if _few_shot_n > 0 else []

    system = build_adaptive_system_prompt(
        "desktop",
        sections=_DESKTOP_SECTIONS,
        task_type="interactive",
        max_entities=max_entities,
        few_shot_examples=_examples or None,
    )
    system = f"{system}\n\n{_DESKTOP_ACTION_HINT}"

    resolved_goal = goal or (instruction.strip().split("\n")[0] if instruction else "")

    extra_parts: list[str] = []
    if instruction:
        extra_parts.append(f"Task instruction: {instruction}")
    if last_action:
        extra_parts.append(f"Last action: {last_action}")
    if last_action_error:
        extra_parts.append(f"Last action error: {last_action_error}")
    if a11y_tree_xml:
        trimmed = a11y_tree_xml.strip()
        if len(trimmed) > 3000:
            trimmed = trimmed[:3000] + "\n…[truncated]"
        extra_parts.append(
            "Accessibility tree (for element grounding, truncated):\n"
            f"{trimmed}"
        )
    if terminal_output:
        tlines = terminal_output.strip().splitlines()
        if len(tlines) > 15:
            tlines = ["…[truncated]"] + tlines[-15:]
        extra_parts.append("Terminal tail:\n" + "\n".join(tlines))

    extra_context = "\n\n".join(extra_parts)

    user_content = build_user_message(
        image,
        domain="desktop",
        task_id=task_id,
        goal=resolved_goal,
        step=step,
        extra_context=extra_context,
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    raw = ""
    schema: str | None = None
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
                    domain="desktop",
                    image_size=image.size if hasattr(image, "size") else None,
                )
                validation = vres.as_dict()
                warnings = warnings + vres.warnings + vres.errors
                break
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
# Convenience: unpack an OSWorld observation dict
# ======================================================================

def osworld_obs_to_schema(
    obs: dict[str, Any],
    *,
    step: int = 0,
    task_id: str = "",
    max_entities: int = 25,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper that unpacks a dict from ``OSWorldGymWrapper``.

    Expected keys: ``screenshot`` (np.ndarray), ``accessibility_tree``
    (XML string), ``terminal`` (string), ``instruction`` (string).
    Missing keys are tolerated — the VLM still receives the screenshot.
    """
    screenshot = obs.get("screenshot")
    if screenshot is None:
        return {
            "schema": None,
            "raw": "",
            "warnings": ["no screenshot"],
            "model": model or _DEFAULT_MODEL,
        }

    if isinstance(screenshot, np.ndarray):
        image = Image.fromarray(screenshot)
    elif isinstance(screenshot, Image.Image):
        image = screenshot
    else:
        return {
            "schema": None,
            "raw": "",
            "warnings": ["unknown screenshot type"],
            "model": model or _DEFAULT_MODEL,
        }

    return generate_label(
        image,
        instruction=obs.get("instruction", ""),
        task_id=task_id,
        step=step,
        a11y_tree_xml=obs.get("accessibility_tree", "") or "",
        terminal_output=obs.get("terminal", "") or "",
        last_action=obs.get("last_action", "") or "",
        last_action_error=obs.get("last_action_error", "") or "",
        max_entities=max_entities,
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
