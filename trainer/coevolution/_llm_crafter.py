"""LLM-driven supplemental crafter (Path 2 / 35B control plane).

Bridges the Phase-1 gap where ``_crafter_hook.run_crafter_step`` only
exercises the deterministic ``crafter.service.SkillCrafterService.reflect_on_episode``
path — that path is rule-based and never touches the 35B-A3B teacher
backbone.  This module sits *alongside* the deterministic Crafter and,
for each ``FailureTrace`` produced by the trainer's per-episode failure
synthesizer, fires one 35B call asking the teacher to propose a
``BankMutationProposal`` (patch / hypothesis / retire) that addresses
that specific failure.

Design (per user decision ``p2_per_failure_trace``):

* **One LLM call per FailureTrace**, capped at ``k_max`` per trainer
  step (default 5).  The cap is a hard slice — failures past the cap
  are dropped, not queued.
* **Parallel** via ``asyncio.gather``.  Each call runs in a thread
  executor (``API_func.ask_model`` is synchronous) so the wall-time
  matches the slowest single 35B call rather than ``k * latency``.
* **Game-schema aware**.  The phase-start :class:`GameProfile` (Path 1)
  is rendered into the system prompt so the teacher sees the same
  ``goal / hazards / key_actions`` context the actor's prompt sees;
  failure traces then anchor to that vocabulary.
* **Fail-soft**.  Every LLM call is wrapped in try/except; parse
  failures, timeouts, and call errors all degrade silently to "no
  proposal from this trace" so a flaky 35B can never take down the
  trainer.

Routing follows :mod:`labeling_supplement._llm_skill_judge`: the model
identifier (default ``BACKBONE_JUDGE_MODEL``) is resolved through
``API_func.ask_model``, which honours ``VLLM_BASE_URL_MAP`` so calls
land on the dedicated 35B endpoint without extra plumbing.

Cross-refs
----------
* ``trainer/coevolution/_crafter_hook.py`` — call site for this module.
* ``trainer/coevolution/_game_schema.py`` — :class:`GameProfile` source.
* ``labeling_supplement/_llm_skill_judge.py`` — same fail-soft pattern.
* ``crafter-harness-orchestrator-roles.md`` §2.1 (Crafter role boundary;
  Path 2 stays inside that boundary — proposals only, no writes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from common.ids import new_proposal_id
from data_structure.extensions.bank_mutation_proposal import (
    BankMutationProposal,
    HypothesisProposal,
    PatchProposal,
    RetireProposal,
)
from data_structure.extensions.failure_trace import FailureTrace
from data_structure.extensions.skill_record import SkillContract

logger = logging.getLogger("trainer.coevolution.llm_crafter")


# ── Tunables ──────────────────────────────────────────────────────────

# Lowered from 5 → 2 after v11 audit. With the prompt rewritten to
# treat ``hypothesize`` as last-resort and the ``existing_skills``
# block injected so the 35B can prefer ``patch``, we want a small
# per-step volume cap as defence-in-depth: even if the teacher
# regresses on prompt-following, we never mint more than 2 new
# hypotheses per game per step. Override via
# ``LLM_CRAFTER_K_MAX`` env var or ``--llm-crafter-k-max`` CLI arg.
DEFAULT_K_MAX_PER_STEP: int = 2
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_TEMPERATURE: float = 0.3
DEFAULT_TIMEOUT_S: float = 60.0
# Hard cap on the number of existing skills we render into the
# ``existing_skills`` prompt block. Keeps the per-call token budget
# bounded for games whose bank has grown large; the most-recently
# added skills are rendered first (they're the most likely match for
# the current failure context).
_MAX_EXISTING_SKILLS_IN_PROMPT: int = 12

# Cap for any free-text field we copy *into* the prompt — keeps token
# budgets bounded for chatty rationales / failed_step bodies.
_MAX_FIELD_CHARS: int = 800
_MAX_PROTOCOL_STEPS: int = 8

# Provenance tag for proposals we mint here.  ``_to_offline_row`` (in
# ``_crafter_hook.py``) uses ``proposer`` to bucket gate decisions, so
# this string lets the dashboard split LLM-driven proposals from the
# rule-based ones.
LLM_CRAFTER_PROPOSER: str = "llm_crafter"


# ── Outcome dataclass ─────────────────────────────────────────────────


@dataclass
class LLMCrafterReport:
    """Summary of one ``run_llm_crafter_async`` call."""

    n_failures_in: int = 0
    n_calls_attempted: int = 0
    n_calls_succeeded: int = 0
    n_calls_failed: int = 0
    n_proposals_emitted: int = 0
    n_parse_failures: int = 0
    n_timeouts: int = 0
    proposals_per_kind: Dict[str, int] = field(default_factory=dict)
    wall_time_s: float = 0.0
    sample_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_failures_in":       self.n_failures_in,
            "n_calls_attempted":   self.n_calls_attempted,
            "n_calls_succeeded":   self.n_calls_succeeded,
            "n_calls_failed":      self.n_calls_failed,
            "n_proposals_emitted": self.n_proposals_emitted,
            "n_parse_failures":    self.n_parse_failures,
            "n_timeouts":          self.n_timeouts,
            "proposals_per_kind":  dict(self.proposals_per_kind),
            "wall_time_s":         self.wall_time_s,
            "sample_errors":       list(self.sample_errors[:5]),
        }


# ── Prompt construction ───────────────────────────────────────────────


def _summarize_failure(failure: FailureTrace) -> Dict[str, Any]:
    """Project one ``FailureTrace`` into a JSON-able prompt payload."""
    return {
        "failure_id":         failure.failure_id,
        "skill_id":           failure.skill_id,
        "skill_episode_id":   failure.skill_episode_id,
        "domain":             failure.domain,
        "failed_step_index":  failure.failed_step_index,
        "failure_class":      failure.failure_class,
        "abort_reason":       (failure.abort_reason or "")[:_MAX_FIELD_CHARS],
        "contract_violation": (failure.contract_violation or "")[:_MAX_FIELD_CHARS],
        "observed_evidence_roles": list(failure.observed_evidence_roles),
        "extra":              {
            k: (str(v)[:_MAX_FIELD_CHARS] if isinstance(v, (str, bytes)) else v)
            for k, v in (failure.extra or {}).items()
        },
    }


def _summarize_game_profile(profile: Any) -> Dict[str, Any]:
    """Compact view of :class:`trainer.coevolution._game_schema.GameProfile`.

    Returns an empty dict if ``profile`` is ``None`` (Path 1 disabled
    on this run).  We only copy the fields the Crafter prompt benefits
    from — the full state-example markup is too verbose for a per-trace
    LLM call.
    """
    if profile is None:
        return {}
    out: Dict[str, Any] = {}
    for fld in (
        "game", "display_name", "genre", "goal", "win_signal",
        "hazards", "recurring_entities", "key_actions", "failure_modes",
    ):
        try:
            v = getattr(profile, fld, None)
        except Exception:
            continue
        if v is not None and v != "" and v != []:
            out[fld] = v
    return out


def _summarize_existing_skills(
    existing_skills: Optional[Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    """Compact view of the bank's current skills for the prompt.

    Each entry is ``{skill_id, name, strategic_description}`` truncated
    to ``_MAX_FIELD_CHARS//4`` so a 12-skill block stays under
    ~600 tokens. The caller is responsible for ordering — we render
    the slice it gives us as-is.
    """
    if not existing_skills:
        return []
    out: List[Dict[str, Any]] = []
    for s in existing_skills[:_MAX_EXISTING_SKILLS_IN_PROMPT]:
        sid = str(s.get("skill_id") or "").strip()
        if not sid:
            continue
        entry = {
            "skill_id": sid,
            "name": str(s.get("name") or sid)[: _MAX_FIELD_CHARS // 4],
        }
        desc = s.get("strategic_description") or s.get("description") or ""
        if desc:
            entry["description"] = str(desc)[: _MAX_FIELD_CHARS // 4]
        out.append(entry)
    return out


def _build_prompt(
    *,
    failure: FailureTrace,
    game: str,
    game_profile: Any = None,
    existing_skills: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    summary = {
        "game":             game,
        "game_profile":     _summarize_game_profile(game_profile),
        "existing_skills":  _summarize_existing_skills(existing_skills),
        "failure":          _summarize_failure(failure),
    }
    summary_json = json.dumps(
        summary, ensure_ascii=False, indent=2, default=str,
    )
    return (
        "You are an offline Crafter for a multi-game RL agent's skill "
        "bank.  A single skill invocation just FAILED in the trainer's "
        "rollout.  Decide what proposal — if any — would best repair "
        "the bank so a future invocation under the same conditions "
        "succeeds.\n"
        "\n"
        "DECISION POLICY (read this first — apply in order):\n"
        "  1. Default to \"none\" when the failure looks like transient\n"
        "     noise (game-over, time-cap, action latency) — a missed\n"
        "     opportunity is cheaper than a polluted bank.\n"
        "  2. Use \"patch\" if ANY entry in `existing_skills` could\n"
        "     plausibly be modified to address this failure.  Look for\n"
        "     tag overlap on skill_id (e.g. COMMIT/NAVIGATE matches a\n"
        "     navigation failure) or token overlap on name /\n"
        "     description.  Patch is the preferred kind whenever a\n"
        "     related skill exists.\n"
        "  3. Use \"retire\" if `failure.skill_id` is set and the skill\n"
        "     is fundamentally broken (contradicts the game's\n"
        "     win_signal, or has been observed to drive the agent into\n"
        "     a death loop).\n"
        "  4. Use \"hypothesize\" — last resort — ONLY when ALL of\n"
        "     these hold simultaneously:\n"
        "       (a) `existing_skills` is empty OR no entry has any\n"
        "           plausible token overlap with the failure context.\n"
        "       (b) You can supply a CONCRETE, GAME-SPECIFIC\n"
        "           `new_skill_name` (e.g. \"AVOID/BULLET_VOLLEY\" not\n"
        "           \"hypothesis_proto\") AND at least 2 SPECIFIC\n"
        "           preconditions that reference game_profile vocabulary\n"
        "           (no boilerplate like \"Action opportunity present\"\n"
        "           or \"Evaluate best available action\").\n"
        "       (c) The protocol has at least 2 steps each referencing\n"
        "           a key_actions verb from `game_profile`.\n"
        "     If any of (a)/(b)/(c) fail, return \"none\" instead.\n"
        "\n"
        "Pick exactly one of these proposal kinds:\n"
        "  - \"patch\":       repair an existing skill (modify protocol\n"
        "                     or contract).  Preferred whenever a\n"
        "                     related skill exists.\n"
        "  - \"retire\":      mark the failed skill for deprecation.\n"
        "                     Use when the skill is incoherent or\n"
        "                     contradicts the game's win_signal.\n"
        "  - \"hypothesize\": LAST RESORT — propose a NEW skill.  Subject\n"
        "                     to the gates listed above; quality bar is\n"
        "                     concrete game-specific content, not a\n"
        "                     generic placeholder.\n"
        "  - \"none\":        no proposal.  This is the right answer\n"
        "                     for transient or uninformative failures.\n"
        "\n"
        "Respond with EXACTLY one JSON object on one or more lines, "
        "and nothing else (no ``` fences, no preamble):\n"
        "  {\n"
        "    \"kind\":          \"patch\" | \"retire\" | \"hypothesize\" | \"none\",\n"
        "    \"rationale\":     \"<one-sentence why>\",\n"
        "    \"target_skill_id\": \"<skill_id to patch/retire, or empty>\",\n"
        "    \"new_skill_name\": \"<new skill name for hypothesize, or empty>\",\n"
        "    \"protocol\":      [\"<step 1 NL>\", \"<step 2 NL>\", ...],\n"
        "    \"preconditions\": [\"<pred>\", ...],\n"
        "    \"effects_add\":   [\"<pred>\", ...],\n"
        "    \"effects_del\":   [\"<pred>\", ...]\n"
        "  }\n"
        "\n"
        "Constraints:\n"
        "  - protocol/preconditions/effects can be empty arrays for\n"
        "    \"retire\" or \"none\".\n"
        "  - Use the game_profile's key_actions vocabulary in protocol\n"
        "    steps when relevant.\n"
        "  - Keep protocol ≤ "
        + str(_MAX_PROTOCOL_STEPS)
        + " steps; longer is wasted budget.\n"
        "\n"
        "INPUT (JSON):\n"
        + summary_json
        + "\n"
    )


# ── Response parsing ──────────────────────────────────────────────────


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(raw: str) -> Optional[Dict[str, Any]]:
    """Pull the first plausible JSON object out of a noisy LLM
    response.  Returns ``None`` if nothing parses."""
    if not raw:
        return None
    text = raw.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if m is not None:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _ensure_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x is not None]
    return [str(v)]


def _wrap_protocol_steps(steps: Sequence[Any]) -> List[Dict[str, Any]]:
    """Convert ``["NL string", ...]`` from the LLM into the typed
    ``[{"action": "EXEC", "payload": {}, "notes": ...}]`` shape that
    ``BankMutationProposal.{patched_protocol|novel_protocol}`` expect
    (mirrors :func:`_crafter_hook._wrap_protocol_steps`).
    """
    out: List[Dict[str, Any]] = []
    for s in steps[:_MAX_PROTOCOL_STEPS]:
        if isinstance(s, dict):
            out.append(dict(s))
        elif isinstance(s, str):
            text = s.strip()
            if text:
                out.append({"action": "EXEC", "payload": {}, "notes": text})
        else:
            text = str(s).strip()
            if text:
                out.append({"action": "EXEC", "payload": {}, "notes": text})
    return out


def _proposal_from_response(
    *,
    parsed: Mapping[str, Any],
    failure: FailureTrace,
    game: str,
    teacher_model: str,
) -> Optional[BankMutationProposal]:
    """Translate one parsed LLM response → typed ``BankMutationProposal``.

    Returns ``None`` on:

    * ``kind == "none"``;
    * unrecognised ``kind``;
    * structural inconsistency (e.g. retire with empty target_skill_id).
    """
    kind = str(parsed.get("kind", "") or "").strip().lower()
    if kind in ("", "none", "skip", "no_op"):
        return None

    rationale = (str(parsed.get("rationale", "") or ""))[:_MAX_FIELD_CHARS]
    target_id = str(parsed.get("target_skill_id", "") or "").strip()
    new_name = str(parsed.get("new_skill_name", "") or "").strip()
    proto_steps = _wrap_protocol_steps(_ensure_str_list(parsed.get("protocol")))
    pre = _ensure_str_list(parsed.get("preconditions"))
    eff_add = _ensure_str_list(parsed.get("effects_add"))
    eff_del = _ensure_str_list(parsed.get("effects_del"))

    # ``BankMutationProposal._ProposalBase`` includes ``parent_skill_ids``
    # / ``seed_failure_ids`` so the gate can crawl provenance.
    parent_ids: List[str] = [target_id] if target_id else []
    if failure.skill_id and failure.skill_id not in parent_ids:
        parent_ids.append(failure.skill_id)
    seed_ids = [failure.failure_id] if failure.failure_id else []

    # All proposals carry a stable ID so the JSONL writer can index
    # them; we reuse the standard mint to stay schema-consistent.
    pid = new_proposal_id()

    contract = SkillContract(
        preconditions=pre,
        effects_add=eff_add,
        effects_del=eff_del,
        expected_evidence_roles=list(failure.observed_evidence_roles or []),
    )

    if kind == "patch":
        # A patch needs a target skill — fall back to the failure's
        # ``skill_id`` if the LLM didn't echo one.
        base_id = target_id or failure.skill_id
        if not base_id:
            return None
        return PatchProposal(
            proposal_id=pid,
            rationale=rationale,
            parent_skill_ids=parent_ids,
            seed_failure_ids=seed_ids,
            target_domains=[failure.domain] if failure.domain else [],
            teacher_model=teacher_model,
            base_skill_id=base_id,
            patched_protocol=proto_steps,
            patched_contract=contract,
            recovery_strategy="llm_crafter_repair",
        )

    if kind == "hypothesize":
        # A hypothesis must produce *something* a downstream gate can
        # evaluate; if both name and protocol are empty the proposal
        # has no signal — drop it.
        name = new_name or f"hyp-{pid[:8]}"
        if not proto_steps and not pre and not eff_add:
            return None
        return HypothesisProposal(
            proposal_id=pid,
            rationale=rationale,
            parent_skill_ids=parent_ids,
            seed_failure_ids=seed_ids,
            target_domains=[failure.domain] if failure.domain else [],
            teacher_model=teacher_model,
            name=name,
            novel_protocol=proto_steps,
            contract=contract,
            source_failure_pattern_ids=seed_ids,
        )

    if kind == "retire":
        target = target_id or failure.skill_id
        if not target:
            return None
        return RetireProposal(
            proposal_id=pid,
            rationale=rationale,
            parent_skill_ids=parent_ids,
            seed_failure_ids=seed_ids,
            target_domains=[failure.domain] if failure.domain else [],
            teacher_model=teacher_model,
            target_skill_id=target,
            reason=rationale or "llm_crafter: retire",
        )

    return None


# ── Single-call worker ────────────────────────────────────────────────


async def _propose_one(
    *,
    failure: FailureTrace,
    game: str,
    game_profile: Any,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    executor: Optional[ThreadPoolExecutor],
    report: LLMCrafterReport,
    enable_thinking: bool = False,
    existing_skills: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Optional[BankMutationProposal]:
    """Run one ``API_func.ask_model`` call → typed proposal.

    All exceptions degrade to ``None`` so ``asyncio.gather`` never sees
    a raised exception.  The ``report`` dataclass is mutated to carry
    the diagnostic counts.

    ``enable_thinking`` (default ``False``) toggles Qwen3 ``<think>``
    reasoning blocks via ``API_func.ask_vllm``'s
    ``chat_template_kwargs={"enable_thinking": ...}``.  Stage 1
    in-domain training keeps it off (fast path); Stage 2 cross-domain
    Crafter callers opt in for higher-quality skill proposals.
    """
    prompt = _build_prompt(
        failure=failure, game=game, game_profile=game_profile,
        existing_skills=existing_skills,
    )
    report.n_calls_attempted += 1
    raw = ""
    try:
        from API_func import ask_model
        from trainer.coevolution._run_loggers import (  # noqa: WPS433
            record_component_call,
        )

        loop = asyncio.get_running_loop()

        def _call() -> str:
            t0 = time.monotonic()
            try:
                return ask_model(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                ) or ""
            finally:
                # Block A5: attribute the wall-clock to the LLM crafter
                # so §6 runtime overhead can split 35B traffic by purpose.
                try:
                    record_component_call(
                        "crafter.llm",
                        latency_ms=(time.monotonic() - t0) * 1000.0,
                    )
                except Exception:  # noqa: BLE001
                    pass

        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(executor, _call),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            report.n_timeouts += 1
            report.n_calls_failed += 1
            report.sample_errors.append(
                f"timeout failure_id={failure.failure_id}"
            )
            return None
    except Exception as exc:  # noqa: BLE001
        report.n_calls_failed += 1
        report.sample_errors.append(
            f"call_error failure_id={failure.failure_id} err={exc!r}"
        )
        logger.debug(
            "llm_crafter: ask_model raised for failure_id=%s err=%s",
            failure.failure_id, exc,
        )
        return None

    parsed = _parse_response(raw)
    if parsed is None:
        report.n_parse_failures += 1
        report.n_calls_failed += 1
        report.sample_errors.append(
            f"parse_failed failure_id={failure.failure_id} "
            f"raw={(raw or '')[:120]!r}"
        )
        logger.debug(
            "llm_crafter: response did not parse for failure_id=%s raw=%r",
            failure.failure_id, (raw or "")[:200],
        )
        return None

    report.n_calls_succeeded += 1
    proposal = _proposal_from_response(
        parsed=parsed, failure=failure, game=game, teacher_model=model,
    )
    if proposal is None:
        return None
    report.n_proposals_emitted += 1
    kind = type(proposal).__name__
    report.proposals_per_kind[kind] = (
        report.proposals_per_kind.get(kind, 0) + 1
    )
    return proposal


# ── Public entrypoint ─────────────────────────────────────────────────


async def run_llm_crafter_async(
    *,
    failures: Sequence[FailureTrace],
    game: str,
    model: str,
    game_profile: Any = None,
    k_max: int = DEFAULT_K_MAX_PER_STEP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    executor: Optional[ThreadPoolExecutor] = None,
    enable_thinking: bool = False,
    existing_skills: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[List[BankMutationProposal], LLMCrafterReport]:
    """Run up to ``k_max`` parallel 35B calls — one per failure trace —
    and return the resulting :class:`BankMutationProposal` list along
    with a summary :class:`LLMCrafterReport`.

    Always returns a value; LLM call failures, timeouts, and parse
    errors are absorbed and surfaced through the report so the caller
    never has to wrap this in try/except.

    Parameters
    ----------
    failures
        Failures to ask the teacher about.  Sliced to the first
        ``k_max``; later ones are dropped (and counted in the report).
    game
        Trainer game name; surfaced in the prompt as context.
    model
        Model identifier passed to ``API_func.ask_model``.  Empty
        string falls back to ``BACKBONE_JUDGE_MODEL`` via
        ``API_func._init_vllm_urls`` (see ``ask_model`` source).
    game_profile
        Optional :class:`trainer.coevolution._game_schema.GameProfile`.
        ``None`` is fine — the prompt just won't carry the profile
        block (Path 1 disabled or fallback).
    k_max
        Hard cap on parallel calls per invocation.
    max_tokens
        Token budget per response.
    temperature
        Sampling temperature.
    timeout_s
        Hard wall-time per single call.
    executor
        Thread pool to run the synchronous ``ask_model`` call on.
        ``None`` uses the asyncio default executor.
    """
    report = LLMCrafterReport()
    report.n_failures_in = len(failures)
    if not failures:
        return [], report

    sliced = list(failures[:k_max])
    if len(failures) > k_max:
        report.sample_errors.append(
            f"sliced {len(failures) - k_max} failures over k_max={k_max}"
        )

    t0 = time.monotonic()
    coros = [
        _propose_one(
            failure=f, game=game, game_profile=game_profile,
            model=model, max_tokens=max_tokens, temperature=temperature,
            timeout_s=timeout_s, executor=executor, report=report,
            enable_thinking=enable_thinking,
            existing_skills=existing_skills,
        )
        for f in sliced
    ]
    # ``return_exceptions=True`` keeps gather from re-raising; but
    # ``_propose_one`` already absorbs all exceptions, so this is just
    # a belt-and-braces guard against future regressions.
    results = await asyncio.gather(*coros, return_exceptions=True)

    proposals: List[BankMutationProposal] = []
    for r in results:
        if isinstance(r, BaseException):
            report.n_calls_failed += 1
            report.sample_errors.append(f"gather_exc: {r!r}")
            continue
        if r is not None:
            proposals.append(r)
    report.wall_time_s = time.monotonic() - t0
    return proposals, report


def run_llm_crafter(
    *,
    failures: Sequence[FailureTrace],
    game: str,
    model: str,
    game_profile: Any = None,
    k_max: int = DEFAULT_K_MAX_PER_STEP,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    executor: Optional[ThreadPoolExecutor] = None,
    enable_thinking: bool = False,
    existing_skills: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[List[BankMutationProposal], LLMCrafterReport]:
    """Synchronous wrapper around :func:`run_llm_crafter_async`.

    Convenience entrypoint for callers that aren't already in an event
    loop AND for callers that ARE inside a running loop (notably
    ``_crafter_hook.run_crafter_step``, which is dispatched
    synchronously from the orchestrator's outer ``co_evolution_loop``
    coroutine).

    Detection is two-stage:

    1. Try ``asyncio.get_running_loop()``.  If it succeeds we're
       inside a running loop and ``asyncio.run`` would raise
       ``RuntimeError("asyncio.run() cannot be called from a running
       event loop")``.  In that case we spawn a worker thread, run a
       fresh event loop there, and join — this isolates the new loop
       from the running one without disturbing it.
    2. Otherwise (no running loop), use ``asyncio.run`` directly.

    The thread-based fallback is the same idiom Jupyter uses for
    nested ``asyncio.run`` calls.  It costs one short-lived thread
    per invocation, which is negligible vs. the ≥ 35B-call latency
    we're already paying.
    """
    coro_args = dict(
        failures=failures, game=game, model=model,
        game_profile=game_profile, k_max=k_max,
        max_tokens=max_tokens, temperature=temperature,
        timeout_s=timeout_s, executor=executor,
        enable_thinking=enable_thinking,
        existing_skills=existing_skills,
    )
    try:
        asyncio.get_running_loop()
        in_running_loop = True
    except RuntimeError:
        in_running_loop = False

    if not in_running_loop:
        return asyncio.run(run_llm_crafter_async(**coro_args))

    import threading
    result_holder: List[Any] = []
    exc_holder: List[BaseException] = []

    def _worker() -> None:
        try:
            result_holder.append(
                asyncio.run(run_llm_crafter_async(**coro_args))
            )
        except BaseException as exc:  # noqa: BLE001
            exc_holder.append(exc)

    t = threading.Thread(target=_worker, name="llm_crafter_runner")
    t.start()
    t.join()
    if exc_holder:
        raise exc_holder[0]
    return result_holder[0]


__all__ = [
    "DEFAULT_K_MAX_PER_STEP",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "LLMCrafterReport",
    "LLM_CRAFTER_PROPOSER",
    "run_llm_crafter",
    "run_llm_crafter_async",
]
