"""Regression locks for ``cold_start/osworld_steering.py`` defaults.

The OSWorld eval at 2026-05-03 surfaced a 400-error path: the steering
helpers (memory summary, reflexion, self-verify) defaulted to
``reasoning_effort='minimal'`` for their cheap text-only LLM calls.
OpenAI's direct ``/v1/chat/completions`` for the gpt-5.x family
hard-rejects ``minimal`` with::

    Unsupported value: 'reasoning_effort' does not support 'minimal'
    with this model. Supported values are: 'none', 'low', 'medium',
    'high', and 'xhigh'.

The fix flipped every default to ``low``. ``low`` is accepted by:

  * OpenAI direct gpt-5.x (the bug case),
  * OpenRouter routes for the same models, and
  * non-reasoning models (Claude / Gemini / Qwen3-VL) — where the
    driver's ``_chat_completion`` silently drops the parameter.

These tests pin those defaults so a future refactor can't quietly
regress to ``minimal`` and break the gpt-5.x direct path again.
"""

from __future__ import annotations

import pytest


def _import_steering_module():
    """Lazy import — keeps the rest of the test suite immune to
    transient breakage if the module's heavy dependencies move."""
    from cold_start import osworld_steering  # noqa: WPS433
    return osworld_steering


# ---------------------------------------------------------------------------
# Per-class defaults
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "class_name",
    ["MemorySummary", "ReflexionTrigger", "SelfVerifier"],
)
def test_steering_dataclass_default_reasoning_effort_is_low(class_name):
    """Every steering dataclass must default ``reasoning_effort`` to
    ``'low'``. Anything else risks the ``minimal`` HTTP-400 regression
    on OpenAI direct gpt-5.x.
    """
    mod = _import_steering_module()
    cls = getattr(mod, class_name)
    fields = cls.__dataclass_fields__
    assert "reasoning_effort" in fields, (
        f"{class_name} must expose a ``reasoning_effort`` field "
        "so callers can override per call."
    )
    default = fields["reasoning_effort"].default
    assert default == "low", (
        f"{class_name}.reasoning_effort default = {default!r}; "
        "must be 'low' (see the May-2026 HTTP-400 regression note in "
        "osworld_steering.py)."
    )


# ---------------------------------------------------------------------------
# Shared helper signature
# ---------------------------------------------------------------------------

def test_shared_helper_default_reasoning_effort_is_low():
    """``_steering_llm_call``'s default keyword argument must mirror
    the dataclass defaults — otherwise a caller using the helper
    directly (e.g. a future steering subsystem) would silently ship
    a non-portable value."""
    import inspect

    mod = _import_steering_module()
    sig = inspect.signature(mod._steering_llm_call)
    param = sig.parameters.get("reasoning_effort")
    assert param is not None, (
        "_steering_llm_call must keep its ``reasoning_effort`` "
        "keyword argument so the caller can override per call."
    )
    assert param.default == "low", (
        f"_steering_llm_call.reasoning_effort default = "
        f"{param.default!r}; must be 'low'."
    )


# ---------------------------------------------------------------------------
# Negative test: 'minimal' specifically must NOT be the default
# ---------------------------------------------------------------------------

def test_minimal_is_not_default_anywhere():
    """Defensive: scan the source for ``reasoning_effort: ... = 'minimal'``
    so a regression caught above is also caught at the source-pattern
    level (in case a future refactor moves the default into a constant
    or factory function and the dataclass-introspection test misses it).
    """
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "cold_start" / "osworld_steering.py"
    )
    text = src.read_text(encoding="utf-8")
    bad_line: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Skip lines that are part of doc-comments / regression notes
        # (those legitimately mention 'minimal' to explain WHY we
        # moved off it).
        if stripped.startswith("#") or stripped.startswith('"'):
            continue
        if (
            "reasoning_effort" in stripped
            and "minimal" in stripped
            and "=" in stripped
            and "default" not in stripped.lower()
        ):
            # Only flag it when 'minimal' is bound as the value (i.e.
            # ``reasoning_effort: ... = "minimal"`` or ``reasoning_effort="minimal"``).
            for needle in ("= 'minimal'", '= "minimal"',
                           "='minimal'", '="minimal"'):
                if needle in stripped:
                    bad_line.append(stripped)
                    break
    assert not bad_line, (
        "found 'reasoning_effort=minimal' default(s) in osworld_steering.py:\n"
        + "\n".join(f"  {b}" for b in bad_line)
        + "\nMove them to 'low' (OpenAI direct gpt-5.x rejects 'minimal')."
    )
