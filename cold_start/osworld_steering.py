"""OSWorld-only opt-in steering modules (Improvements #3, #4, #6).

This module is *additive*: nothing in it runs unless the corresponding
``--enable_memory`` / ``--enable_reflection`` / ``--enable_self_verify``
flag is passed to ``cold_start/generate_cold_start_actor_osworld.py``.

Why a separate module:
----------------------
The memory-summary, reflection, and self-verification logic is OSWorld-
specific (it depends on AT-SPI / pyautogui / desktop-task semantics)
and each call costs an extra LLM round-trip. Other corpora
(browsergym, gymv, visual-toolbench, …) share the same cold-start
driver template but should never see this code path. By isolating
the three subsystems behind a thin facade in this file, the OSWorld
actor's main loop stays identical when the flags are off:

    if memory is not None:                # only if --enable_memory
        memory_block = memory.maybe_refresh(...)
    if reflector is not None:             # only if --enable_reflection
        reflection_block = reflector.maybe_reflect(...)
    if verifier is not None and action == "DONE":  # only if --enable_self_verify
        if not verifier.verify_done(...):
            action = ...                  # downgrade DONE → continue

Each subsystem is a small dataclass with a single public method; they
do not share state and can be enabled independently.

The shared LLM-call helper :func:`_steering_llm_call` is used by both
the memory and reflection paths to avoid pulling the schema-VLM stack
into this module — the caller passes a pre-built ``client`` and the
chosen routed model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared LLM helper (text-only). Vision is intentionally NOT used here —
# self-verifier vision is wired through the same ``_chat_completion`` the
# main loop uses; see SelfVerifier below.
# ---------------------------------------------------------------------------

def _steering_llm_call(
    *,
    client: Any,
    routed_model: str,
    chat_completion_fn: Callable[..., Any],
    system: str,
    user: str,
    max_tokens: int = 350,
    reasoning_effort: Optional[str] = "low",
    temperature: float = 0.0,
) -> Tuple[Optional[str], Optional[str]]:
    """Cheap text-only LLM call for memory / reflection summaries.

    Returns ``(content, error)``. Uses ``reasoning_effort=low`` by
    default — these calls are summarisation, not planning, and burning
    hidden-thinking tokens on them defeats the cost gating. Caller can
    override per-call.

    Note: we used to default to ``minimal`` here, but OpenAI's direct
    /v1/chat/completions for gpt-5.4 hard-rejects that value with
    ``HTTP 400 Unsupported value: 'reasoning_effort' does not support
    'minimal' with this model. Supported values are: 'none', 'low',
    'medium', 'high', and 'xhigh'.``  ``low`` is accepted by every
    OpenAI reasoning model AND silently ignored by non-reasoning
    models (Claude / Gemini / Qwen3-VL on OpenRouter), so it is the
    portable default.
    """
    try:
        resp = chat_completion_fn(
            client,
            model=routed_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        return content or None, None
    except Exception as exc:  # noqa: BLE001 - log + degrade
        logger.warning("[steering] LLM call failed: %s", exc)
        return None, repr(exc)


# ---------------------------------------------------------------------------
# Improvement #3: Step-level memory summary
# ---------------------------------------------------------------------------

_MEMORY_SYSTEM_PROMPT = (
    "You are the memory summariser for an OSWorld desktop agent. The "
    "actor LLM is about to choose its next action and has already "
    "consumed too many recent steps to remember reliably. Produce a "
    "concise running summary (≤6 short bullets) covering:\n"
    "  • What sub-goals have been COMPLETED so far (concrete + visible).\n"
    "  • Which subgoals were ATTEMPTED but did NOT advance (dead ends).\n"
    "  • Any APP STATE the actor must remember (open dialogs, "
    "    cursor position, selected file, current document path).\n"
    "  • What the next reasonable subgoal is.\n"
    "Do NOT include speculative claims. Do NOT mention raw pixel "
    "coordinates. Be brief — the actor only needs a memory aid, not "
    "a re-narration."
)


@dataclass
class MemorySummary:
    """Maintains a running summary refreshed every ``refresh_every`` steps.

    Activated only when ``--enable_memory`` is passed. The summary text
    is injected verbatim into the actor's user prompt as a
    ``<memory>...</memory>`` block.
    """

    client: Any
    routed_model: str
    chat_completion_fn: Callable[..., Any]
    refresh_every: int = 5
    max_tokens: int = 350
    reasoning_effort: Optional[str] = "low"
    last_summary: Optional[str] = None
    last_refresh_step: int = -1
    refresh_count: int = 0
    refresh_failures: int = 0

    def maybe_refresh(
        self,
        *,
        step: int,
        task: str,
        history: List[Dict[str, Any]],
        subgoals: List[str],
    ) -> Optional[str]:
        """Return the (possibly updated) memory block to inject.

        Refresh fires when:
          • this is the first call (step >= refresh_every), AND
          • it has been at least ``refresh_every`` steps since the
            previous successful refresh.

        Returns the cached summary on non-refresh steps so the actor
        always sees the most recent memory, never a stale one.
        """
        if step < self.refresh_every:
            return None
        if step - self.last_refresh_step < self.refresh_every:
            return self.last_summary
        if not history and not subgoals:
            return None

        # Build a compact, deterministic input — recent action / reward
        # / subgoal triples newest-last. We cap at the last
        # ``2 * refresh_every`` steps to bound token spend.
        cap = max(self.refresh_every * 2, 10)
        recent = list(zip(
            subgoals[-cap:] if subgoals else [""] * len(history),
            history[-cap:] if history else [],
        ))
        lines: List[str] = []
        for i, (sg, h) in enumerate(recent):
            a = (h.get("action") if h else "") or ""
            r = (h.get("reward") if h else 0.0) or 0.0
            err = bool(h.get("error")) if h else False
            noop = bool(h.get("noop")) if h else False
            tag = []
            if err:
                tag.append("ERR")
            if noop:
                tag.append("noop")
            tag_s = f" [{','.join(tag)}]" if tag else ""
            lines.append(
                f"  {i+1:>2}. subgoal={sg!r:<40} "
                f"action={a[:60]!r} reward={r:+.2f}{tag_s}"
            )
        user = (
            f"Task instruction:\n  {task}\n\n"
            f"Step now: {step}\n\n"
            f"Recent step trail (newest last):\n" + "\n".join(lines)
        )
        summary, err = _steering_llm_call(
            client=self.client,
            routed_model=self.routed_model,
            chat_completion_fn=self.chat_completion_fn,
            system=_MEMORY_SYSTEM_PROMPT,
            user=user,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        if summary:
            self.last_summary = summary
            self.last_refresh_step = step
            self.refresh_count += 1
        else:
            self.refresh_failures += 1
        return self.last_summary

    def stats(self) -> Dict[str, Any]:
        return {
            "refresh_count": self.refresh_count,
            "refresh_failures": self.refresh_failures,
            "last_refresh_step": self.last_refresh_step,
            "has_summary": self.last_summary is not None,
        }


# ---------------------------------------------------------------------------
# Improvement #4: Reflexion-lite — fire on consecutive no-op streaks
# ---------------------------------------------------------------------------

_REFLECTION_SYSTEM_PROMPT = (
    "You are a debugging coach for an OSWorld desktop agent that has "
    "produced N consecutive no-op steps (the action did NOT change "
    "the screen and produced NO error). Diagnose WHY in one short "
    "paragraph (≤4 sentences) and PROPOSE EXACTLY 3 alternative "
    "actions. Format strictly as:\n"
    "  Diagnosis: <one short paragraph>\n"
    "  Alternatives:\n"
    "    1. <pyautogui-style action or hotkey>\n"
    "    2. <pyautogui-style action or hotkey>\n"
    "    3. <pyautogui-style action or hotkey>\n"
    "Each alternative must be a CONCRETE action string the agent can "
    "copy verbatim — for example "
    "``pyautogui.hotkey('alt','f')`` or "
    "``click_element(id=12)`` or ``pyautogui.click(440, 612)``. "
    "Do NOT propose vague advice like 'try harder' or 'wait'."
)


@dataclass
class ReflexionTrigger:
    """Fires a one-shot reflection LLM call on consecutive no-op streaks.

    Activated only when ``--enable_reflection`` is passed. The result
    is injected as a ``<reflection>...</reflection>`` block into the
    NEXT step's actor prompt. Per-streak — does not re-fire until the
    no-op streak is broken.
    """

    client: Any
    routed_model: str
    chat_completion_fn: Callable[..., Any]
    trigger_streak: int = 2
    cooldown_steps: int = 4
    max_tokens: int = 400
    reasoning_effort: Optional[str] = "low"
    last_reflection: Optional[str] = None
    last_fired_step: int = -10_000
    fire_count: int = 0
    fire_failures: int = 0
    _consumed_for_step: int = -1

    def maybe_reflect(
        self,
        *,
        step: int,
        consecutive_noops: int,
        last_action: str,
        task: str,
        recent_subgoals: List[str],
        recent_history: List[Dict[str, Any]],
    ) -> Optional[str]:
        """If conditions hold, produce a reflection block; else None.

        Conditions:
          • ``consecutive_noops >= trigger_streak``
          • cooldown elapsed since the last fire
        """
        if consecutive_noops < self.trigger_streak:
            return None
        if step - self.last_fired_step < self.cooldown_steps:
            return None

        recent_lines: List[str] = []
        for sg, h in zip(recent_subgoals[-5:], recent_history[-5:]):
            a = (h.get("action") or "") if h else ""
            recent_lines.append(f"  - subgoal={sg!r}, action={a[:80]!r}")
        user = (
            f"Task: {task}\n"
            f"Step now: {step}\n"
            f"The previous {consecutive_noops} steps were no-ops "
            f"(no state change, no error). The last action was:\n"
            f"  {last_action!r}\n\n"
            f"Recent (subgoal, action) trail:\n"
            + ("\n".join(recent_lines) if recent_lines else "  (none)")
        )
        reflection, err = _steering_llm_call(
            client=self.client,
            routed_model=self.routed_model,
            chat_completion_fn=self.chat_completion_fn,
            system=_REFLECTION_SYSTEM_PROMPT,
            user=user,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
        )
        if reflection:
            self.last_reflection = reflection
            self.last_fired_step = step
            self.fire_count += 1
        else:
            self.fire_failures += 1
        return self.last_reflection

    def consume_for(self, step: int) -> Optional[str]:
        """Return the reflection block to render this step, then clear it.

        The reflection should appear on the user prompt EXACTLY ONCE
        (the step the agent recovers on); after that the actor either
        succeeds or triggers a new reflection on the next streak.
        """
        if self.last_reflection is None:
            return None
        if self._consumed_for_step == step:
            return None
        self._consumed_for_step = step
        block = self.last_reflection
        self.last_reflection = None  # one-shot
        return block

    def stats(self) -> Dict[str, Any]:
        return {
            "fire_count": self.fire_count,
            "fire_failures": self.fire_failures,
            "last_fired_step": self.last_fired_step,
        }


# ---------------------------------------------------------------------------
# Improvement #6: Self-verification on DONE
# ---------------------------------------------------------------------------

_VERIFY_SYSTEM_PROMPT = (
    "You are an OSWorld evaluator. The actor agent just emitted "
    "``DONE`` claiming the task is complete. Look at the screenshot "
    "(passed below) and the task instruction, and decide STRICTLY:\n"
    "  • Reply with ONLY the single token ``YES`` if the on-screen "
    "    state objectively satisfies the user's instruction.\n"
    "  • Reply with ONLY the single token ``NO`` followed by one "
    "    short reason if not.\n"
    "Do NOT speculate about hidden / off-screen / saved file state. "
    "Do NOT trust progress claims from the actor. Look ONLY at what "
    "is visible in the screenshot. If the screenshot is ambiguous "
    "(e.g. dialog still open, file save not confirmed), reply NO."
)


@dataclass
class SelfVerifier:
    """Vision-grounded gate before accepting a DONE emission.

    Activated only when ``--enable_self_verify`` is passed. The
    actor's DONE only commits if this verifier returns YES; otherwise
    the action is rewritten to a no-op-ish ``WAIT`` so the loop
    continues and the actor gets one more chance to actually solve
    the task.
    """

    client: Any
    routed_model: str
    chat_completion_fn: Callable[..., Any]
    max_tokens: int = 60
    reasoning_effort: Optional[str] = "low"
    verify_count: int = 0
    verify_yes: int = 0
    verify_no: int = 0
    verify_failures: int = 0
    last_decision: Optional[str] = None
    last_reason: Optional[str] = None

    def verify_done(
        self,
        *,
        task: str,
        screenshot_data_url: Optional[str],
    ) -> Tuple[bool, str]:
        """Return ``(is_actually_done, reason_text)``.

        Falls open (returns True) if the verification call itself
        fails — we should not penalise the actor for a transient API
        outage. The fallback decision is logged so it is auditable.
        """
        self.verify_count += 1
        if not screenshot_data_url:
            self.verify_failures += 1
            self.last_decision = "fallback_yes_no_image"
            return True, "no_screenshot"
        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Task instruction:\n  {task}\n\n"
                    "Look at the screenshot and decide YES / NO."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": screenshot_data_url},
            },
        ]
        try:
            resp = self.chat_completion_fn(
                self.client,
                model=self.routed_model,
                messages=[
                    {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
            content = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[self-verify] LLM call failed: %s", exc)
            self.verify_failures += 1
            self.last_decision = "fallback_yes_on_error"
            self.last_reason = repr(exc)
            return True, "verify_call_failed"

        first_token = content.split()[0].upper() if content else ""
        if first_token.startswith("YES"):
            self.verify_yes += 1
            self.last_decision = "yes"
            self.last_reason = content[:200]
            return True, content[:200]
        if first_token.startswith("NO"):
            self.verify_no += 1
            self.last_decision = "no"
            self.last_reason = content[:200]
            return False, content[:200]
        # Unparseable response — fall open to avoid blocking legitimate
        # DONEs but log so it shows up in the run audit.
        logger.info("[self-verify] unparseable response: %r", content[:80])
        self.verify_failures += 1
        self.last_decision = "fallback_yes_unparseable"
        self.last_reason = content[:200]
        return True, "unparseable"

    def stats(self) -> Dict[str, Any]:
        return {
            "verify_count": self.verify_count,
            "verify_yes": self.verify_yes,
            "verify_no": self.verify_no,
            "verify_failures": self.verify_failures,
            "last_decision": self.last_decision,
        }


__all__ = [
    "MemorySummary",
    "ReflexionTrigger",
    "SelfVerifier",
]
