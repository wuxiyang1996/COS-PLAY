"""LLM-driven post-validation pass for the trainer's harness (Path 4).

Bridges the Phase-1 gap where ``_harness_hook.SkillHarnessHook.validate_choice``
runs only the deterministic ``SkillHarness.validate_invocation`` —
that path is rule-based and never touches the 35B-A3B teacher
backbone.  This module sits *behind* the deterministic validator and,
when policy says to fire, asks the teacher to render a verdict on
the same ``(skill, state)`` pair so a SHADOW or contract-translated
admit can be downgraded to a veto when the LLM disagrees.

Hybrid policy (per user decision ``agree_hybrid_bootstrap``):

* **Bootstrap window** — when the trainer's outer step is below
  ``bootstrap_steps`` (default 20) the validator fires on EVERY
  admitted skill regardless of deterministic certainty.  Early in
  training the deterministic veto is unreliable (cold-start bank,
  first invocations), so the LLM acts as a co-validator.
* **Steady state** — the validator only fires when the deterministic
  verdict was "uncertain": SHADOW status, no can_handle evidence,
  translation-rewritten contract, or a generic fallback flag.
* **One-way downgrade** — the LLM verdict can ONLY change admit→veto.
  It can never override a deterministic veto upward.  This makes the
  layered system a strict tightening, not a contradiction.
* **Episode-level cache** — verdicts are keyed by
  ``(episode_id, skill_id)`` so repeated picks of the same skill
  inside one episode pay the LLM cost only once.

Routing follows :mod:`labeling_supplement._llm_skill_judge`: the
model identifier (default ``BACKBONE_JUDGE_MODEL``) is resolved
through ``API_func.ask_model``, which honours ``VLLM_BASE_URL_MAP``.

Cross-refs
----------
* ``trainer/coevolution/_harness_hook.py`` — call site for this module.
* ``labeling_supplement/_llm_skill_judge.py`` — same fail-soft pattern.
* ``harness/README.md`` §22 (trainer integration block).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from data_structure.extensions.skill_record import SkillRecord

logger = logging.getLogger("trainer.coevolution.llm_harness_validator")


# ── Tunables ──────────────────────────────────────────────────────────

DEFAULT_BOOTSTRAP_STEPS: int = 20
DEFAULT_MAX_TOKENS: int = 256
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_TIMEOUT_S: float = 30.0

_MAX_FIELD_CHARS: int = 600
_MAX_PROTOCOL_STEPS: int = 8

# Verdict tag we stamp into the diagnostic dict so dashboards can
# split LLM-vetoed picks from deterministic-vetoed ones.
LLM_VALIDATOR_TAG: str = "llm_harness_validator"


# ── Outcome dataclass ─────────────────────────────────────────────────


@dataclass
class LLMValidatorOutcome:
    """One LLM Harness validator call result."""

    ok: bool
    rationale: str = ""
    raw_response: str = ""
    error: Optional[str] = None
    cache_hit: bool = False
    fired: bool = True
    skip_reason: Optional[str] = None  # populated when ``fired=False``

    def to_diag(self) -> Dict[str, Any]:
        return {
            LLM_VALIDATOR_TAG: {
                "ok":         self.ok,
                "rationale":  self.rationale,
                "fired":      self.fired,
                "cache_hit":  self.cache_hit,
                "skip_reason": self.skip_reason,
                "error":      self.error,
            },
        }


@dataclass
class LLMValidatorStats:
    """Per-step roll-up of validator activity, mirrored into
    :class:`HarnessStepStats` via ``LLMHarnessValidator.stats``."""

    n_calls_attempted: int = 0
    n_calls_succeeded: int = 0
    n_calls_failed: int = 0
    n_parse_failures: int = 0
    n_timeouts: int = 0
    n_cache_hits: int = 0
    n_skipped_steady: int = 0       # skipped: steady-state, deterministic certain
    n_skipped_no_record: int = 0    # skipped: SkillRecord not in cache
    n_admit_overrides: int = 0      # admit→veto downgrades
    n_admit_confirmed: int = 0      # admit→admit (LLM agrees)
    last_errors: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_calls_attempted":   self.n_calls_attempted,
            "n_calls_succeeded":   self.n_calls_succeeded,
            "n_calls_failed":      self.n_calls_failed,
            "n_parse_failures":    self.n_parse_failures,
            "n_timeouts":          self.n_timeouts,
            "n_cache_hits":        self.n_cache_hits,
            "n_skipped_steady":    self.n_skipped_steady,
            "n_skipped_no_record": self.n_skipped_no_record,
            "n_admit_overrides":   self.n_admit_overrides,
            "n_admit_confirmed":   self.n_admit_confirmed,
            "last_errors":         list(self.last_errors[:5]),
        }


# ── Uncertainty heuristic ─────────────────────────────────────────────


def is_uncertain_admit(
    *,
    skill: SkillRecord,
    deterministic_diag: Mapping[str, Any],
) -> bool:
    """Decide whether the deterministic admit was 'uncertain' enough
    to warrant a 35B second look in steady state.

    Uncertainty signals (any one suffices):

    1. SHADOW skill — admitted only because ``allow_shadow=True``.
    2. Predicate-translated contract — the harness ran the source
       record through the predicate translator before admitting.
       (We can't read the translation status from the
       ``deterministic_diag`` directly because that's emitted at
       ``filter_candidates`` time, not ``validate_choice`` time, but
       the caller can pass it forward via the ``bindings`` dict if
       needed.  Conservative default below.)
    3. Adapter rejection: ``deterministic_diag.get("status") == "ok"``
       with empty ``can_handle_evidence`` — admit-by-default.
    4. Fallback / degraded paths flagged via
       ``deterministic_diag.get("degraded") == True``.
    """
    status = getattr(skill, "status", None)
    status_value = getattr(status, "value", None) or str(status or "")
    if status_value.lower() == "shadow":
        return True
    if deterministic_diag.get("degraded"):
        return True
    handle_ev = deterministic_diag.get("can_handle_evidence")
    if handle_ev is None or (isinstance(handle_ev, (list, dict)) and not handle_ev):
        return True
    if deterministic_diag.get("translation_status") in ("rewritten", "failed"):
        return True
    return False


# ── Prompt construction ───────────────────────────────────────────────


def _summarize_skill(skill: SkillRecord) -> Dict[str, Any]:
    """Compact view of the SkillRecord — same shape as the LLM-judge
    summarizer but trimmed for the per-step validator's tighter
    token budget."""
    contract = getattr(skill, "contract", None)
    return {
        "skill_id":   getattr(skill, "skill_id", ""),
        "name":       getattr(skill, "name", ""),
        "skill_type": getattr(getattr(skill, "skill_type", None), "value", "") or "",
        "status":     getattr(getattr(skill, "status", None), "value", "") or "",
        "source_domains":   list(getattr(skill, "source_domains", []) or []),
        "feasible_domains": list(getattr(skill, "feasible_domains", []) or []),
        "protocol":   (getattr(skill, "protocol", []) or [])[:_MAX_PROTOCOL_STEPS],
        "contract": {
            "preconditions":  list(getattr(contract, "preconditions", []) or [])[:6],
            "effects_add":    list(getattr(contract, "effects_add", []) or [])[:6],
            "effects_del":    list(getattr(contract, "effects_del", []) or [])[:6],
            "expected_evidence_roles": list(
                getattr(contract, "expected_evidence_roles", []) or [],
            ),
        } if contract is not None else {},
    }


def _summarize_state(state: Any) -> Dict[str, Any]:
    """Extract the actionable bits of a :class:`StateSchema` for the
    validator prompt.  ``state`` is duck-typed — we accept anything
    that has ``domain`` / ``task`` / ``extra``."""
    extra = getattr(state, "extra", None) or {}
    if not isinstance(extra, dict):
        extra = {}
    return {
        "domain":         getattr(state, "domain", ""),
        "task":           getattr(state, "task", ""),
        "summary_state":  str(extra.get("summary_state", ""))[:_MAX_FIELD_CHARS],
        "intention":      str(extra.get("intention", ""))[:200],
    }


def _summarize_game_profile(profile: Any) -> Dict[str, Any]:
    """Compact view of :class:`trainer.coevolution._game_schema.GameProfile`.

    Returns an empty dict when profile is None or missing (Path 1
    disabled).  Only the cheap-to-render fields are copied — the
    Path-1 ``state_example_markup`` is too verbose for a per-step
    validator call.
    """
    if profile is None:
        return {}
    out: Dict[str, Any] = {}
    for fld in (
        "game", "display_name", "genre", "goal", "win_signal",
        "hazards", "key_actions", "failure_modes",
    ):
        try:
            v = getattr(profile, fld, None)
        except Exception:
            continue
        if v is not None and v != "" and v != []:
            out[fld] = v
    return out


def _build_prompt(
    *,
    skill: SkillRecord,
    state: Any,
    game_profile: Any = None,
    deterministic_diag: Mapping[str, Any],
) -> str:
    summary = {
        "game_profile":        _summarize_game_profile(game_profile),
        "state":               _summarize_state(state),
        "skill":               _summarize_skill(skill),
        "deterministic_admit": dict(deterministic_diag) if deterministic_diag else {},
    }
    summary_json = json.dumps(
        summary, ensure_ascii=False, indent=2, default=str,
    )
    return (
        "You are a second-pass Harness validator for a multi-game RL "
        "agent.  The deterministic harness has just ADMITTED the "
        "following skill for invocation in the given state.  Decide "
        "whether the admit should STAND or whether the skill should "
        "be VETOED based on coherence between skill, state, and the "
        "game's win condition.\n"
        "\n"
        "Veto criteria (any one suffices):\n"
        "  - The skill's preconditions clearly contradict the current "
        "state (e.g. skill needs an enemy on screen, state has none).\n"
        "  - The skill's effects move the agent away from the game's "
        "win_signal (e.g. defensive skill when goal=offensive push).\n"
        "  - The skill's protocol uses actions outside the game's "
        "key_actions vocabulary.\n"
        "  - The skill is otherwise mis-bound (wrong domain "
        "translation, wrong task axis, nonsensical for this state).\n"
        "\n"
        "Otherwise, KEEP the admit — do not veto on style / "
        "preference / minor inefficiency.\n"
        "\n"
        "Respond with EXACTLY one JSON object on a single line, with "
        "these fields and nothing else:\n"
        "  {\"verdict\": \"keep\" | \"veto\",\n"
        "   \"reason\":  \"<one short sentence>\"}\n"
        "\n"
        "INPUT (JSON):\n"
        + summary_json
        + "\n"
    )


# ── Response parsing ──────────────────────────────────────────────────


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_verdict(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Pull ``(verdict, reason)`` out of a noisy LLM response.

    Returns ``(None, None)`` only if no plausible verdict token can
    be recovered.
    """
    if not raw:
        return (None, None)
    txt = raw.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            v = obj.get("verdict")
            r = obj.get("reason", "")
            if isinstance(v, str):
                return (v.strip().lower(), str(r).strip())
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(txt)
    if m is not None:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                v = obj.get("verdict")
                r = obj.get("reason", "")
                if isinstance(v, str):
                    return (v.strip().lower(), str(r).strip())
        except Exception:
            pass
    low = txt.lower()
    # Prefer the more-conservative token when both appear (so a model
    # that hedges with "veto unless..." still vetoes).
    if "veto" in low or "reject" in low:
        return ("veto", "")
    if "keep" in low or "admit" in low or "ok" in low:
        return ("keep", "")
    return (None, None)


# ── Validator class with per-episode cache ────────────────────────────


class LLMHarnessValidator:
    """Stateful per-step LLM validator.

    Constructed once per :class:`SkillHarnessHook` (and therefore once
    per ``(game, trainer_step)``), reused across every episode in
    that step.  Carries the episode-level cache and aggregate stats.

    The cache is keyed by ``(episode_id, skill_id)`` so repeatedly
    binding the same skill inside one episode pays the LLM cost
    exactly once.  Cache entries from a *different* episode in the
    same step are NOT cleared — the assumption is that repeat picks
    across episodes carry the same evidence (state may differ but
    the skill's action vocabulary doesn't), which is the default the
    user signed off on with ``agree_hybrid_bootstrap``.

    Lifecycle methods:

    * :meth:`should_fire` — pure-Python policy decision.  ``True``
      when bootstrap window OR uncertain admit OR cache miss.
    * :meth:`validate` — runs the LLM call (or returns the cached
      verdict), absorbs all errors, returns
      :class:`LLMValidatorOutcome`.
    """

    def __init__(
        self,
        *,
        model: str,
        trainer_step: int,
        bootstrap_steps: int = DEFAULT_BOOTSTRAP_STEPS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        game_profile: Any = None,
    ) -> None:
        self._model = model
        self._trainer_step = int(trainer_step)
        self._bootstrap_steps = int(bootstrap_steps)
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._timeout_s = float(timeout_s)
        self._game_profile = game_profile

        # cache[ (episode_id, skill_id) ] = LLMValidatorOutcome
        self._cache: Dict[tuple, LLMValidatorOutcome] = {}
        self._stats = LLMValidatorStats()

    @property
    def stats(self) -> LLMValidatorStats:
        return self._stats

    @property
    def trainer_step(self) -> int:
        return self._trainer_step

    @property
    def bootstrap_steps(self) -> int:
        return self._bootstrap_steps

    def in_bootstrap(self) -> bool:
        """True when the trainer step is inside the bootstrap window
        and the validator should fire on every admit."""
        return self._trainer_step < self._bootstrap_steps

    def should_fire(
        self,
        *,
        skill: SkillRecord,
        deterministic_diag: Mapping[str, Any],
    ) -> bool:
        """Policy gate: bootstrap-window OR uncertain admit."""
        if self.in_bootstrap():
            return True
        return is_uncertain_admit(
            skill=skill, deterministic_diag=deterministic_diag,
        )

    # ── Synchronous validate (called from inside ``validate_choice``) ─

    def validate(
        self,
        *,
        episode_id: str,
        skill: SkillRecord,
        state: Any,
        deterministic_diag: Mapping[str, Any],
    ) -> LLMValidatorOutcome:
        """Run one validator pass, honouring policy + cache.

        Always returns a value; LLM call failures, timeouts, and
        parse errors all degrade to a ``keep`` outcome with a
        diagnostic ``error`` field set, so the caller's
        ``validate_choice`` never has to wrap this in try/except.
        """
        skill_id = getattr(skill, "skill_id", "") or ""
        cache_key = (episode_id or "", skill_id)

        if not self.should_fire(skill=skill, deterministic_diag=deterministic_diag):
            self._stats.n_skipped_steady += 1
            return LLMValidatorOutcome(
                ok=True, fired=False, skip_reason="steady_state_certain",
            )

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._stats.n_cache_hits += 1
            # Re-stamp the cache flag so the caller's diagnostic dict
            # reflects the cache hit (the original ``cached`` object
            # was minted with cache_hit=False on first call).
            return LLMValidatorOutcome(
                ok=cached.ok,
                rationale=cached.rationale,
                raw_response=cached.raw_response,
                error=cached.error,
                cache_hit=True,
                fired=cached.fired,
                skip_reason=cached.skip_reason,
            )

        self._stats.n_calls_attempted += 1
        prompt = _build_prompt(
            skill=skill, state=state,
            game_profile=self._game_profile,
            deterministic_diag=deterministic_diag,
        )
        raw = ""
        try:
            from API_func import ask_model
            from trainer.coevolution._run_loggers import (  # noqa: WPS433
                record_component_call,
            )
            t0 = time.monotonic()
            raw = ask_model(
                prompt,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ) or ""
            elapsed = time.monotonic() - t0
            try:
                record_component_call(
                    "harness.validator",
                    latency_ms=elapsed * 1000.0,
                )
            except Exception:  # noqa: BLE001
                pass
            if elapsed > self._timeout_s:
                # ``API_func.ask_model`` is synchronous and doesn't
                # carry a timeout knob; we observe the wall-time and
                # treat overruns as soft timeouts.  The verdict still
                # gets considered, but we tag the stats so dashboards
                # know.
                self._stats.n_timeouts += 1
        except Exception as exc:  # noqa: BLE001
            self._stats.n_calls_failed += 1
            self._stats.last_errors.append(
                f"call_error skill={skill_id} err={exc!r}"
            )
            logger.debug(
                "llm_harness_validator: ask_model raised for skill=%s err=%s",
                skill_id, exc,
            )
            outcome = LLMValidatorOutcome(
                ok=True,  # one-way: errors degrade to keep
                rationale="llm_validator: call_failed (kept admit)",
                raw_response="",
                error=str(exc),
                cache_hit=False,
                fired=True,
            )
            self._cache[cache_key] = outcome
            self._stats.n_admit_confirmed += 1  # error == kept admit
            return outcome

        verdict, reason = _parse_verdict(raw)
        if verdict is None:
            self._stats.n_parse_failures += 1
            self._stats.n_calls_failed += 1
            self._stats.last_errors.append(
                f"parse_failed skill={skill_id} raw={(raw or '')[:120]!r}"
            )
            logger.debug(
                "llm_harness_validator: response did not parse for "
                "skill=%s raw=%r",
                skill_id, (raw or "")[:200],
            )
            outcome = LLMValidatorOutcome(
                ok=True,
                rationale="llm_validator: parse_failed (kept admit)",
                raw_response=raw,
                error="parse_failed",
                cache_hit=False,
                fired=True,
            )
            self._cache[cache_key] = outcome
            self._stats.n_admit_confirmed += 1
            return outcome

        self._stats.n_calls_succeeded += 1
        if verdict == "veto":
            self._stats.n_admit_overrides += 1
            outcome = LLMValidatorOutcome(
                ok=False,
                rationale=(reason or "llm: veto")[:_MAX_FIELD_CHARS],
                raw_response=raw,
                cache_hit=False,
                fired=True,
            )
            logger.info(
                "llm_harness_validator: VETO skill=%s episode=%s "
                "reason=%r (trainer_step=%d, bootstrap=%s)",
                skill_id, episode_id, outcome.rationale[:80],
                self._trainer_step, self.in_bootstrap(),
            )
        else:
            self._stats.n_admit_confirmed += 1
            outcome = LLMValidatorOutcome(
                ok=True,
                rationale=(reason or "llm: keep")[:_MAX_FIELD_CHARS],
                raw_response=raw,
                cache_hit=False,
                fired=True,
            )
            logger.info(
                "llm_harness_validator: KEEP skill=%s episode=%s "
                "(trainer_step=%d, bootstrap=%s)",
                skill_id, episode_id, self._trainer_step,
                self.in_bootstrap(),
            )
        self._cache[cache_key] = outcome
        return outcome


__all__ = [
    "DEFAULT_BOOTSTRAP_STEPS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "LLMHarnessValidator",
    "LLMValidatorOutcome",
    "LLMValidatorStats",
    "LLM_VALIDATOR_TAG",
    "is_uncertain_admit",
]
