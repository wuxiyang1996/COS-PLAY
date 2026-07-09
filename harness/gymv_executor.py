"""Gymv hop executor — bind `GymvAdapter` to a real Gymnasium-style env.

The default `_deterministic_executor` in `harness/adapters/gymv_adapter.py`
is a stub for unit-tests and the gate's dry-run path. This module is the
real wiring: given an env handle (anything that exposes a Gymnasium-shape
`reset()` / `step(action_str) -> (obs, reward, term, trunc, info)` —
including `env_wrappers.gym_like.make_gaming_env(...)` and the gymv
multi-agent envs from `cold_start/generate_cold_start_actor_gymv.py`), it
returns a `HopExecutor` callable suitable for `GymvAdapter.set_executor`.

PLAN-HARNESS §22 (Day-3 cross-game adaptation): the executor maps each
typed hop (`{op, payload, effects_add, …}` from the protocol lift) onto
a concrete env action string, runs `env.step`, parses the resulting
schema_canonical (or text observation) into a fresh `StateSchema`, and
exposes that to the per-hop success_fn so effect predicates can be
evaluated against pre/post snapshots.

Env-mutating verbs (SLIDE / MOVE / ROTATE / DROP / PLACE / SELECT / SWAP /
APPROACH / EXECUTE) translate into env actions via `ACTION_ALIAS_MAP`
(taxonomy → game-specific token list, picked up by the env's
`action_names`). Observational verbs (INSPECT / READ / TRACK / COMPARE /
EVALUATE / SIMULATE / VERIFY / KEEP / STOP / CONTINUE) do *not* step the
env — they only synthesise an `EvidenceRef` so the harness's G0 invariant
(non-empty evidence on success for non-ACTION skills) holds.

The executor is a closure factory: callers build an env once, hand it
in, and reuse the resulting `HopExecutor` across multiple
`harness.run_skill(...)` invocations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from common.state_schema import EvidenceRef, StateSchema
from harness.adapters.gymv_adapter import HopExecutor
from harness.gym_schema_producer import SchemaProducer
from harness.skill_adapter import AdapterRunContext

logger = logging.getLogger("harness.gymv_executor")


def _adapt_schema_producer(
    producer: SchemaProducer,
    *,
    domain: str,
    task: str,
) -> Callable[[Any, Mapping[str, Any]], Optional[str]]:
    """Adapt a Day-4 `SchemaProducer` (`(info, obs, *, step, task, goal)
    -> str`) into the legacy `schema_builder` signature
    (`(obs, info) -> Optional[str]`) the executor's internal call sites
    expect. Domain/task/step/goal are closed over at adapter time so the
    producer keeps a clean signature even when the executor doesn't own
    those values."""

    state = {"step": 0}  # mutable counter so each call advances

    def _builder(observation: Any, info: Mapping[str, Any]) -> Optional[str]:
        try:
            block = producer(info, observation, step=state["step"], task=task)
        except Exception as exc:                                        # noqa: BLE001
            logger.debug(
                "schema_producer raised (%s); reverting to text obs", exc,
            )
            return None
        state["step"] += 1
        return block or None

    return _builder


# ---------------------------------------------------------------------------
# Action alias map: protocol-lift verb → env-side action token list.
#
# The first entry whose token appears in `env.action_names` wins. This
# lets one alias map handle several games (e.g. SLIDE.up == "up" for 2048
# and "left" for Tetris-style left-shifts when the env exposes one and
# not the other).
# ---------------------------------------------------------------------------

# `OP -> {payload-key: {payload-value -> [candidate env tokens]}}`
ACTION_ALIAS_MAP: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "SLIDE": {
        "direction": {
            "up":    ["up", "north", "U"],
            "down":  ["down", "south", "D"],
            "left":  ["left", "west", "L"],
            "right": ["right", "east", "R"],
        },
    },
    "MOVE": {
        "direction": {
            "up":    ["up", "north", "U"],
            "down":  ["down", "south", "D"],
            "left":  ["left", "west", "L"],
            "right": ["right", "east", "R"],
        },
    },
    "ROTATE": {
        "dir": {
            "cw":  ["rotate_cw", "rotate_right", "spin_cw", "rotate"],
            "ccw": ["rotate_ccw", "rotate_left", "spin_ccw"],
        },
        "direction": {
            "cw":  ["rotate_cw", "rotate_right", "spin_cw", "rotate"],
            "ccw": ["rotate_ccw", "rotate_left", "spin_ccw"],
        },
    },
    "DROP": {},     # bare "drop"-ish action when env exposes one
    "PLACE": {},
    "EXECUTE": {},  # fallback merge / clear / commit step
    "SELECT": {},
    "SWAP": {},
    "APPROACH": {},
}

# Bare op-level fallbacks when the payload doesn't pin the action.
_OP_FALLBACK_TOKENS: Dict[str, List[str]] = {
    "DROP":    ["drop", "hard_drop", "down", "land"],
    "PLACE":   ["drop", "place"],
    "EXECUTE": ["merge", "submit", "clear", "noop"],
    "SELECT":  ["select", "click"],
    "SWAP":    ["swap"],
    "APPROACH":[],
}

# Verbs that observe / reason but never step the env. The hop is still
# recorded (with an EvidenceRef), the env state stays put.
OBSERVATIONAL_OPS: frozenset[str] = frozenset({
    "INSPECT", "READ", "TRACK",
    "COMPARE", "EVALUATE", "SIMULATE", "PREFER", "PENALIZE", "VERIFY",
    "KEEP", "STOP", "CONTINUE",
})

ENV_MUTATING_OPS: frozenset[str] = frozenset({
    "SLIDE", "MOVE", "ROTATE", "DROP", "PLACE",
    "SELECT", "SWAP", "EXECUTE", "APPROACH",
})


class _GymStyleEnv(Protocol):
    """Structural type covering the env shapes we accept.

    Both `env_wrappers.gym_like._GymLikeWrapper.step(action_str)` and
    raw `gym.Env.step(action)` satisfy this — we only require a 5-tuple
    return value with a dict-shaped or string-shaped observation and a
    list of valid action names exposed via `action_names` (preferred)
    or `info["action_names"]` (fallback).
    """

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]: ...
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Tuple[Any, Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Action resolution
# ---------------------------------------------------------------------------

def _resolve_action(
    op: str,
    payload: Mapping[str, Any],
    *,
    action_names: Sequence[str],
    alias_map: Mapping[str, Mapping[str, Mapping[str, List[str]]]] = ACTION_ALIAS_MAP,
) -> Optional[str]:
    """Pick the env-side action string for `(op, payload)` or return None.

    Precedence (most → least specific):
      1. `payload["action"]` is a string in `action_names`.
      2. `alias_map[op][slot][payload[slot]]` resolves to a token in
         `action_names`. Handles `SLIDE.direction → "up"` etc.
      3. ANY string value in `payload` matches an entry in
         `action_names` directly. This is the load-bearing rescue clause
         for cold-start lifted skills where the actor binds a directional
         token into a slot we didn't statically alias (e.g. the lifted
         COMMIT/MERGE protocol uses `SELECT.target=${target}` and the
         actor populates `${target}="up"`; the env exposes `"up"` as a
         direct action so we should just use it).
      4. `_OP_FALLBACK_TOKENS[op]` first matching token (DROP, EXECUTE).
      5. None — caller treats as `abort_reason="no_env_action"`.
    """

    if not action_names:
        return None
    names = list(action_names)
    name_set = {n.lower(): n for n in names}

    # 1. Explicit `action` slot in payload.
    raw = payload.get("action") or payload.get("env_action")
    if isinstance(raw, str) and raw.lower() in name_set:
        return name_set[raw.lower()]

    # 2. Aliased payload slots.
    for slot_name, mapping in (alias_map.get(op) or {}).items():
        slot_val = payload.get(slot_name)
        if not isinstance(slot_val, str):
            continue
        candidates = mapping.get(slot_val.lower())
        if not candidates:
            continue
        for cand in candidates:
            if cand.lower() in name_set:
                return name_set[cand.lower()]

    # 3. Direct payload-value rescue: any payload value that is already
    # a known env action. Skips obvious placeholder tokens (`${slot}`).
    for v in payload.values():
        if not isinstance(v, str):
            continue
        if v.startswith("${") and v.endswith("}"):
            continue
        if v.lower() in name_set:
            return name_set[v.lower()]

    # 4. Op-level fallback tokens (DROP, EXECUTE, …).
    for cand in _OP_FALLBACK_TOKENS.get(op, ()):
        if cand.lower() in name_set:
            return name_set[cand.lower()]

    return None


# ---------------------------------------------------------------------------
# Observation → StateSchema
# ---------------------------------------------------------------------------

def _coerce_to_text(observation: Any) -> str:
    """Pull a textual representation out of an env observation.

    Handles:
      * `env_wrappers.gym_like` dict obs (`{"text": ..., "img_path": ...}`).
      * `gym_v.core.Observation` (has `.text`).
      * Plain strings.
    """

    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    if isinstance(observation, Mapping):
        for key in ("text", "schema_canonical", "textual_representation"):
            v = observation.get(key)
            if isinstance(v, str) and v:
                return v
    text = getattr(observation, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _state_from_env_obs(
    *,
    observation: Any,
    info: Mapping[str, Any],
    domain: str,
    task: str,
    cumulative_reward: float,
    terminated: bool,
    truncated: bool,
    inner_step: int,
    outer_step: int,
    schema_builder: Optional[Callable[[Any, Mapping[str, Any]], Optional[str]]] = None,
) -> StateSchema:
    """Build a runtime `StateSchema` from a Gymnasium-style observation.

    Strategy:
      * If `schema_builder` is provided AND it returns a non-empty
        canonical schema string, defer to `parse_schema_canonical` for
        the full structured StateSchema (rich `entity_attrs`, etc.).
      * Else, fall back to a minimal schema with the cumulative reward
        surfaced as `facts["score"]` and the terminal flag as
        `facts["phase"]`. This is enough for the
        `cumulative_reward_increased` / `phase_transitioned` predicates
        to evaluate even when the env doesn't emit a canonical block.
    """

    canonical: Optional[str] = None
    if schema_builder is not None:
        try:
            canonical = schema_builder(observation, info)
        except Exception as exc:                                   # noqa: BLE001
            logger.debug("schema_builder raised (%s); falling back to text obs", exc)
            canonical = None

    if canonical is None:
        text = _coerce_to_text(observation)
        if "<state>" in text and "</state>" in text:
            canonical = text

    if canonical:
        # Local import to avoid circular import (helpers depend on
        # `common.state_schema.StateSchema`, this module imports neither
        # transitively).
        from labeling_supplement._harness_io_helpers import parse_schema_canonical
        s = parse_schema_canonical(canonical, default_domain=domain)
        # Always overlay cumulative reward + terminal flag — the env's
        # `info["score"]` is the source of truth for transient evaluation
        # even when the canonical block also carries `score`.
        if cumulative_reward is not None:
            s.facts.setdefault("cumulative_reward", float(cumulative_reward))
        s.facts["phase"] = (
            "gameover" if terminated else
            "truncated" if truncated else
            (s.facts.get("phase") or "play")
        )
        s.facts["terminated"] = bool(terminated)
        s.facts["truncated"] = bool(truncated)
        s.task = task or s.task
        s.inner_step = inner_step
        s.outer_step = outer_step
        return s

    # Fallback: synthesise a minimal facts dict from the env's text obs.
    text = _coerce_to_text(observation)
    return StateSchema(
        task=task,
        domain=domain,
        elements=[],
        facts={
            "score": float(cumulative_reward) if cumulative_reward is not None else 0.0,
            "cumulative_reward": float(cumulative_reward or 0.0),
            "phase": "gameover" if terminated else "truncated" if truncated else "play",
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "raw_text": text[:2000] if text else "",
        },
        inner_step=inner_step,
        outer_step=outer_step,
        extra={"parser": "fallback_text_obs"},
    )


# ---------------------------------------------------------------------------
# The factory itself
# ---------------------------------------------------------------------------

@dataclass
class GymvExecutorState:
    """Mutable state the executor closure carries across hops."""

    cumulative_reward: float = 0.0
    last_obs: Any = None
    last_info: Dict[str, Any] = None  # type: ignore[assignment]
    terminated: bool = False
    truncated: bool = False
    outer_step: int = 0
    last_post_state: Optional[StateSchema] = None


def initial_state_from_env(
    env: _GymStyleEnv,
    *,
    domain: str = "gymv",
    task: str = "",
    schema_builder: Optional[Callable[[Any, Mapping[str, Any]], Optional[str]]] = None,
    schema_producer: Optional[SchemaProducer] = None,
    seed: Optional[int] = None,
) -> StateSchema:
    """Build a `StateSchema` from the env's *current* (post-reset) observation.

    The harness's `run_skill(skill, state)` requires a `state` argument
    that the success_fn will treat as the hop-0 pre-state. For real env
    runs that means the pre-state must mirror the env's actual board so
    `cumulative_reward_increased` / `entity_value_increased` predicates
    have a baseline. Using a fresh `StateSchema(task=..., domain=...)`
    produces empty `facts`, which forces every predicate into the
    *undecidable* branch.

    Call this once per episode, immediately after `env.reset()`:

        env = make_gaming_env("twenty_forty_eight")
        env.reset()
        state = initial_state_from_env(env, task="twenty_forty_eight")
        episode = harness.run_skill(skill, state, parent_run_id=None)

    `seed` is forwarded to `env.reset(seed=...)` when set; we never call
    reset implicitly so the caller stays in control of episode framing.
    """

    if seed is not None:
        env.reset(seed=seed)
    obs = getattr(env, "_last_obs", None)
    info: Dict[str, Any] = {}
    if obs is None:
        # Fall back to a fresh reset — the env has not been observed.
        obs, info = env.reset()
    builder = schema_builder
    if builder is None and schema_producer is not None:
        builder = _adapt_schema_producer(
            schema_producer, domain=domain, task=task,
        )
    return _state_from_env_obs(
        observation=obs,
        info=info,
        domain=domain,
        task=task,
        cumulative_reward=0.0,
        terminated=False,
        truncated=False,
        inner_step=0,
        outer_step=0,
        schema_builder=builder,
    )


def make_gymv_executor(
    env: _GymStyleEnv,
    *,
    domain: str = "gymv",
    task: Optional[str] = None,
    schema_builder: Optional[Callable[[Any, Mapping[str, Any]], Optional[str]]] = None,
    schema_producer: Optional[SchemaProducer] = None,
    action_names: Optional[Sequence[str]] = None,
    alias_map: Mapping[str, Mapping[str, Mapping[str, List[str]]]] = ACTION_ALIAS_MAP,
    state_holder: Optional[GymvExecutorState] = None,
    on_unresolved: str = "skip",
) -> Tuple[HopExecutor, GymvExecutorState]:
    """Bind `env` into a `HopExecutor` for `GymvAdapter.set_executor`.

    Args:
      env: A Gymnasium-style env. `env_wrappers.gym_like.make_gaming_env`
        works out of the box; raw gym/gym-v envs work as long as they
        expose an `action_names` attribute or surface one through
        `info["action_names"]`.
      domain: The domain tag projected onto each post-step `StateSchema`.
      task: Optional override for `state.task`. Defaults to whatever
        appears in the canonical schema text (or the empty string).
      schema_builder: Optional callable mapping `(obs, info)` to a
        full canonical `<state>...</state>` string. If omitted, we look
        for `obs["text"]` / `obs["schema_canonical"]`; if neither carries
        a canonical block, the executor falls back to a minimal facts
        dict (still enough for `cumulative_reward_increased` /
        `phase_transitioned`).
      action_names: Pre-resolved env action vocabulary. When None, we
        consult `env.action_names` then `info["action_names"]` from the
        most-recent step (the gymv `_GymLikeWrapper` writes this).
      alias_map: See `ACTION_ALIAS_MAP`.
      state_holder: Optional shared `GymvExecutorState` so the caller
        can read out `cumulative_reward`, `last_obs`, etc. between runs
        without re-binding the env.
      on_unresolved: What to do when an env-mutating op (SLIDE / MOVE /
        EXECUTE / SELECT / …) cannot resolve a token in `action_names`:

        * ``"skip"`` (default) — emit a GATHER EvidenceRef with a
          ``"no_env_action"`` payload and continue. This is the right
          behaviour for the lifted COMMIT/MERGE protocol where the
          verbose ``EXECUTE()`` hop is redundant w.r.t. a preceding
          ``SLIDE.direction=up`` — the env was already stepped, so
          turning the second hop into a no-op preserves the scored run.
        * ``"abort"`` — return ``ok=False`` with a structured
          ``no_env_action_for_op`` reason. Use this when you want every
          unresolvable hop to surface as a failure (e.g. to find missing
          slot bindings during gate hardening).

    Returns: `(executor, state_holder)`. The state holder is mutated in
    place across calls.
    """

    if on_unresolved not in {"skip", "abort"}:
        raise ValueError(
            f"on_unresolved={on_unresolved!r} must be 'skip' or 'abort'"
        )
    holder = state_holder or GymvExecutorState(last_info={})

    # If the caller passed a Day-4 SchemaProducer, adapt it to the
    # legacy schema_builder shape. Explicit `schema_builder` always
    # wins over `schema_producer` (caller knows best).
    builder: Optional[Callable[[Any, Mapping[str, Any]], Optional[str]]]
    if schema_builder is not None:
        builder = schema_builder
    elif schema_producer is not None:
        builder = _adapt_schema_producer(
            schema_producer, domain=domain, task=task or "",
        )
    else:
        builder = None

    def _action_vocab() -> List[str]:
        if action_names:
            return list(action_names)
        names = getattr(env, "action_names", None)
        if names:
            return list(names)
        info = holder.last_info or {}
        return list(info.get("action_names") or [])

    def executor(action_type: str, payload: Dict[str, Any], ctx: AdapterRunContext) -> Dict[str, Any]:
        op = (action_type or "STEP").upper()

        # Observational hop — no env step, just an EvidenceRef.
        if op in OBSERVATIONAL_OPS:
            holder.last_post_state = ctx.state
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload),
                    "no_env_step": True,
                },
                "evidence": [
                    EvidenceRef(
                        source=f"gymv:{op.lower()}",
                        locator=f"step={ctx.state.inner_step}",
                        role="GATHER",
                        confidence=0.9,
                    )
                ],
                "post_state": ctx.state.to_json(),
            }

        # Env-mutating hop — resolve and step.
        names = _action_vocab()
        action_str = _resolve_action(op, payload, action_names=names,
                                     alias_map=alias_map)
        if action_str is None:
            reason = (
                f"no_env_action_for_op={op}_payload={dict(payload)} "
                f"action_names={names}"
            )
            if on_unresolved == "abort":
                return {"ok": False, "reason": reason, "evidence": []}
            # Soft-skip: record the hop as observational evidence so the
            # success_fn isn't blocked by a redundant verb (e.g. EXECUTE
            # following SLIDE).
            holder.last_post_state = ctx.state
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload),
                    "no_env_step": True,
                    "skip_reason": reason,
                },
                "evidence": [
                    EvidenceRef(
                        source=f"gymv:{op.lower()}",
                        locator=f"step={ctx.state.inner_step},skip",
                        role="GATHER",
                        confidence=0.5,
                        payload={"reason": "no_env_action"},
                    )
                ],
                "post_state": ctx.state.to_json(),
            }

        t0 = time.time()
        try:
            obs, reward, terminated, truncated, info = env.step(action_str)
        except Exception as exc:                                   # noqa: BLE001
            return {
                "ok": False,
                "reason": f"env_step_raised: {exc!r}",
                "evidence": [],
            }
        elapsed_ms = (time.time() - t0) * 1000.0

        holder.cumulative_reward += float(reward or 0.0)
        holder.last_obs = obs
        holder.last_info = dict(info or {})
        holder.terminated = bool(terminated)
        holder.truncated = bool(truncated)
        holder.outer_step += 1

        post_state = _state_from_env_obs(
            observation=obs,
            info=info or {},
            domain=domain,
            task=task or ctx.state.task,
            cumulative_reward=holder.cumulative_reward,
            terminated=terminated,
            truncated=truncated,
            inner_step=ctx.state.inner_step + 1,
            outer_step=holder.outer_step,
            schema_builder=builder,
        )
        holder.last_post_state = post_state

        evidence = [
            EvidenceRef(
                source=f"gymv:{op.lower()}",
                locator=f"step={holder.outer_step},action={action_str}",
                role="COMMIT",
                confidence=1.0,
                payload={
                    "reward": float(reward or 0.0),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                },
            )
        ]

        # An env step that ended the episode terminally is NOT an
        # adapter-level failure — the harness's outcome distinguishes
        # "skill ran" from "skill achieved its effect", and the success_fn
        # decides the latter. We only fail the hop when the env raised.
        return {
            "ok": True,
            "observation": {
                "action_str": action_str,
                "reward": float(reward or 0.0),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "elapsed_ms": elapsed_ms,
            },
            "evidence": evidence,
            "post_state": post_state.to_json(),
        }

    return executor, holder


__all__ = [
    "ACTION_ALIAS_MAP",
    "ENV_MUTATING_OPS",
    "GymvExecutorState",
    "OBSERVATIONAL_OPS",
    "initial_state_from_env",
    "make_gymv_executor",
]
