"""GPT-4o-driven Actor that gathers per-step SFT data.

This is the "data collection" flavour of
:class:`decision_agents.actor_agent.ActorAgent`.  It inherits the full
schema-native pipeline (parse → intention → reselect → inner-MDP →
action prompt → entity resolution → anti-repetition) and overrides
exactly two seams:

1. :meth:`_call_llm` — sends the action prompt as a chat-completion
   with optional ``image_url`` content parts so the SFT teacher
   (``gpt-5.5``) sees the same screenshot the
   ``Qwen/Qwen3.5-9B`` student will.
2. :meth:`step` — after the parent populates the
   :class:`~decision_agents.actor_agent.ActorDecision`, it writes a
   per-step row through :class:`~decision_agents.SFT.sft_recorder.SFTRecorder`
   in the format :mod:`trainer.SFT.data_loader` already understands.

Why subclass instead of compose
-------------------------------
The legacy :class:`ActorAgent` already encodes every contract from
``plans/02-action-agent/PLAN-ACTION-AGENT.md`` (slot-coverage,
inner-MDP, reselect-on-stall, anti-repetition, reward-aware hop
costs).  Re-implementing those in a sibling class would mean two
sources of truth.  Subclassing keeps the contract delta minimal: one
swapped LLM call, one new artefact written.

Vision routing
--------------
GPT-4o is reachable through ``API_func.ask_gpt`` either directly via
OpenAI or via OpenRouter (default in this repo).  Both endpoints accept
the OpenAI ``chat.completions`` content-part format.  We use the
caller-provided OpenAI client when available and fall back to the text
path (``ask_model``) when the caller passes no images, so callers that
haven't wired up vision yet still get a working actor.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from decision_agents.actor_agent import (
    ActorAgent,
    ActorDecision,
    DEFAULT_MODEL as _LEGACY_DEFAULT_MODEL,
)
from decision_agents.core.harness import Harness
from decision_agents.core.multimodal import (
    VisualInput,
    build_openai_vision_messages,
)
from decision_agents.reward_func import RewardConfig
from decision_agents.schema_parser import StateSchema
from decision_agents.skill_interface import SkillProvider
from decision_agents.SFT.sft_recorder import SFTRecorder

try:
    from API_func import ask_gpt as _ask_gpt
except ImportError:  # pragma: no cover — optional dep for offline tests
    _ask_gpt = None

try:  # OpenAI SDK is the multimodal path
    import openai as _openai
except ImportError:  # pragma: no cover
    _openai = None

_LOGGER = logging.getLogger(__name__)

try:
    from common.models import BACKBONE_SFT_TEACHER_MODEL as _SFT_TEACHER_MODEL
except Exception:  # pragma: no cover
    _SFT_TEACHER_MODEL = "gpt-5.5"

DEFAULT_GPT4O_MODEL = _SFT_TEACHER_MODEL
"""SFT collection uses ``BACKBONE_SFT_TEACHER_MODEL`` (``gpt-5.5`` by
default — see ``common/models.py``).  The constant name
``DEFAULT_GPT4O_MODEL`` is preserved for backward compatibility with
callers that import it; cheaper variants (``gpt-5.5-mini`` etc.) tend
to degrade label quality on the schema + action-taking tasks.
Override only via the constructor *model* arg."""

# Default vision system prompt.  Kept short — the per-step
# action prompt already contains the schema, valid actions, skill
# block, and inner-MDP scratchpad, so the system prompt only has to
# nudge the model into the strict ``SUBGOAL/REASONING/ACTION`` format
# the trainer/SFT loader's `_align_action_taking_to_coevolution`
# expects.
DEFAULT_VISION_SYSTEM_PROMPT = (
    "You are an Actor Agent for a multimodal task. You will be shown a "
    "screenshot together with a structured state summary and a list of "
    "valid environment actions. Choose exactly ONE valid action.\n"
    "\n"
    "Output STRICTLY in this format (no extra prose):\n"
    "SUBGOAL: [TAG] <your immediate objective in <=15 words>\n"
    "REASONING: <1-2 sentences citing what you saw in the screenshot>\n"
    "ACTION: <one valid action, copied verbatim, or its 1-based number>"
)


# ──────────────────────────────────────────────────────────────────────
# GPT4oCollectorActor
# ──────────────────────────────────────────────────────────────────────


class GPT4oCollectorActor(ActorAgent):
    """Actor wired to GPT-4o that records SFT data per step.

    Parameters (in addition to :class:`ActorAgent`)
    -----------------------------------------------
    recorder
        Optional :class:`SFTRecorder`.  When ``None`` no records are
        written — useful for offline tests that just want the actor's
        behaviour without filesystem side effects.
    game
        Game / domain identifier used as the subdirectory under the
        recorder's ``output_dir``.  Matches the per-game layout
        ``trainer.SFT.config.SFTConfig`` reads.
    vision_system_prompt
        Override for the chat-completion system prompt.  Defaults to
        :data:`DEFAULT_VISION_SYSTEM_PROMPT`.
    openai_client
        Pre-built ``openai.OpenAI`` client.  When ``None`` the actor
        constructs one against ``OPENAI_API_KEY`` (or routes via
        OpenRouter when ``open_router_api_key`` is wired into
        ``API_func``).  Unit tests can pass a mock client to avoid
        network calls without monkeypatching globals.

    Notes
    -----
    The *images* attached to a step persist on the actor for the
    duration of one outer step and are automatically forwarded to the
    overridden :meth:`_call_llm`.  Pass them via :meth:`step` (the
    optional ``images`` keyword), or set them with
    :meth:`set_step_images` from a runner that builds frames lazily.
    """

    def __init__(
        self,
        *,
        recorder: Optional[SFTRecorder] = None,
        game: str = "unknown",
        vision_system_prompt: str = DEFAULT_VISION_SYSTEM_PROMPT,
        openai_client: Optional[Any] = None,
        skill_provider: Optional[SkillProvider] = None,
        harness: Optional[Harness] = None,
        reward_config: Optional[RewardConfig] = None,
        model: str = DEFAULT_GPT4O_MODEL,
        stall_window: int = 4,
        # Deprecated kwargs forwarded to ActorAgent (which warns + ignores).
        hop_policy: Any = None,
        max_hops_per_step: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            skill_provider=skill_provider,
            harness=harness,
            reward_config=reward_config,
            stall_window=stall_window,
            hop_policy=hop_policy,
            max_hops_per_step=max_hops_per_step,
        )
        self.recorder = recorder
        self.game = game
        self.vision_system_prompt = vision_system_prompt
        self._openai_client = openai_client
        self._step_images: List[VisualInput] = []
        # Cached on every _pick_action call so the post-step record can
        # log the exact prompt the LLM saw without re-rendering.
        self._last_prompt: str = ""
        self._last_reply: str = ""

    # ── per-step image plumbing ──────────────────────────────────────

    def set_step_images(self, images: Optional[Sequence[VisualInput]]) -> None:
        """Stage images for the next :meth:`step` call."""
        self._step_images = list(images) if images else []

    def step(  # type: ignore[override]
        self,
        *,
        observation: str,
        schema_text: Optional[str] = None,
        schema: Optional[StateSchema] = None,
        task: str = "",
        valid_actions: Optional[List[str]] = None,
        info: Optional[Dict[str, Any]] = None,
        images: Optional[Sequence[VisualInput]] = None,
    ) -> ActorDecision:
        """Run one outer step and persist an SFT record.

        Same contract as :meth:`ActorAgent.step` plus an optional
        *images* keyword; when supplied, the screenshots are forwarded
        to GPT-4o and recorded in the JSONL row's ``image`` block.
        """
        if images is not None:
            self.set_step_images(images)

        decision = super().step(
            observation=observation,
            schema_text=schema_text,
            schema=schema,
            task=task,
            valid_actions=valid_actions,
            info=info,
        )

        if self.recorder is not None and self._last_prompt and self._last_reply:
            try:
                self.recorder.record_action_taking(
                    prompt=self._last_prompt,
                    completion=self._last_reply,
                    game=self.game,
                    intention=decision.intention or "",
                    active_skill=decision.active_skill_id or "",
                    image=self._step_images[0] if self._step_images else None,
                    extras={
                        "parse_path": decision.parse_path,
                        "valid_actions": list(decision.valid_actions),
                    },
                )
            except Exception as exc:
                _LOGGER.warning("SFT recorder failed for step: %s", exc)

        # Drop staged images once the step closes so the next step
        # starts clean.  Callers that want sticky images can re-stage
        # via set_step_images() before the next step().
        self._step_images = []
        self._last_prompt = ""
        self._last_reply = ""
        return decision

    # ── LLM seam override (vision-aware) ─────────────────────────────

    def _call_llm(
        self,
        prompt: str,
        *,
        images: Optional[List[Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 200,
    ) -> str:
        """Send *prompt* (and the staged screenshots) to GPT-4o.

        Routing:

        * with images → OpenAI chat completions with content parts
          (``[text, image_url, ...]``);
        * without images → fall back to ``API_func.ask_gpt`` so the
          collector keeps working in pure-text rollouts.
        """
        # Cache for the post-step recorder.
        self._last_prompt = prompt

        attached: List[VisualInput] = list(images) if images else list(self._step_images)

        if not attached:
            reply = self._text_only_call(prompt, temperature, max_tokens)
            self._last_reply = reply
            return reply

        try:
            client = self._get_openai_client()
        except Exception as exc:
            _LOGGER.warning(
                "OpenAI client unavailable; falling back to text path: %s", exc
            )
            reply = self._text_only_call(prompt, temperature, max_tokens)
            self._last_reply = reply
            return reply

        messages = build_openai_vision_messages(
            prompt=prompt,
            images=attached,
            system=self.vision_system_prompt,
        )
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = resp.choices[0].message.content if resp.choices else ""
        except Exception as exc:
            _LOGGER.warning("GPT-4o vision call failed: %s", exc)
            reply = ""

        self._last_reply = reply or ""
        return self._last_reply

    # ── private ───────────────────────────────────────────────────────

    def _text_only_call(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Text-only fallback using :func:`API_func.ask_gpt`.

        Kept on the collector (not on the base actor) because we want
        to ensure the SFT data-collection path always uses GPT-4o
        regardless of the parent class's ``ask_model`` shortcut.
        """
        if _ask_gpt is None:
            return super()._call_llm(
                prompt, temperature=temperature, max_tokens=max_tokens
            )
        try:
            return _ask_gpt(
                prompt,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
            ) or ""
        except Exception as exc:
            _LOGGER.warning("ask_gpt call failed: %s", exc)
            return ""

    def _get_openai_client(self) -> Any:
        if self._openai_client is not None:
            return self._openai_client
        if _openai is None:
            raise RuntimeError(
                "openai SDK not installed; cannot run GPT-4o vision path"
            )
        # Prefer OpenRouter when the project key is wired (matches the
        # ``API_func.ask_gpt`` routing rule).  Otherwise use direct
        # OpenAI.  Done lazily so unit tests that pass `openai_client=...`
        # never touch env state.
        try:
            from API_func import (
                open_router_api_key,
                openai_api_key,
                OPENROUTER_BASE,
            )
        except Exception:
            open_router_api_key = os.environ.get("OPENROUTER_API_KEY") or ""
            openai_api_key = os.environ.get("OPENAI_API_KEY") or ""
            OPENROUTER_BASE = "https://openrouter.ai/api/v1"

        if open_router_api_key and open_router_api_key.strip():
            client = _openai.OpenAI(
                base_url=OPENROUTER_BASE, api_key=open_router_api_key.strip()
            )
        else:
            if not openai_api_key:
                raise RuntimeError(
                    "neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set"
                )
            client = _openai.OpenAI(api_key=openai_api_key)
        self._openai_client = client
        return client


__all__ = [
    "GPT4oCollectorActor",
    "DEFAULT_GPT4O_MODEL",
    "DEFAULT_VISION_SYSTEM_PROMPT",
]
