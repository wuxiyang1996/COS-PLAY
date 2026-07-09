"""Gymv success-function — predicate evaluation against pre/post `StateSchema`.

Day-3 of the cross-game adaptation roadmap (harness/README §22). The
protocol lift (`labeling/_protocol_lift.py`) emits typed effect predicates
per env-mutating hop: `entity_value_increased`, `cumulative_reward_increased`,
`phase_transitioned`, `entity_count_changed`, `entity_appeared`,
`entity_disappeared`, `attribute_changed`, `entity_value_decreased`. This
module turns those predicates into runtime checks against consecutive
`metadata.schema_canonical` snapshots, so the harness can decide whether a
real env step *actually* did what the skill claimed.

It is the runtime counterpart to the **mining** logic in
`labeling/_protocol_lift.py`. Predicate `type` strings come from the same
taxonomy (`EFFECT_PREDICATE_TYPES` there); when the lift adds a new
predicate type, this module gains a corresponding evaluator branch.

Flow (PLAN-HARNESS §5.4 + §22):

    pre  = parse_schema_canonical(...)         # before env.step(action)
    post = parse_schema_canonical(...)         # after env.step(action)
    res  = evaluate_hop_effects(hop, pre, post)
    # res.passed:   True  ⇔ every predicate in hop["effects_add"] held
    # res.violated: list of effect-predicate dicts that were violated
    # res.skipped:  predicates we couldn't decide (e.g. label not in schema)

The default scorer (`per_step_success_fn`) walks the per-step
`pre_state` / `post_state` snapshots that `GymvAdapter.run` records and
computes pass-rate over the env-mutating hops. The orchestrator then
plugs this into `FewShotAdapter(success_fn=...)` for Stage 3a transfer
verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from common.state_schema import StateSchema
from data_structure.extensions.skill_episode import SkillEpisode

# Re-export the predicate taxonomy so callers don't have to import from the
# labeling package (keeps the runtime side oblivious to label-side packaging).
EFFECT_PREDICATE_TYPES: Tuple[str, ...] = (
    "entity_value_increased",
    "entity_value_decreased",
    "entity_count_changed",
    "entity_appeared",
    "entity_disappeared",
    "attribute_changed",
    "cumulative_reward_increased",
    "phase_transitioned",
)


# ---------------------------------------------------------------------------
# Core predicate evaluator
# ---------------------------------------------------------------------------

@dataclass
class PredicateResult:
    """Outcome of one predicate evaluation."""

    predicate: Dict[str, Any]
    passed: Optional[bool]   # None ⇒ undecidable (missing fields)
    detail: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": str(self.predicate.get("type")),
            "args": dict(self.predicate.get("args") or {}),
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class HopEffectResult:
    """Per-hop roll-up of effect-predicate evaluations."""

    hop_index: int
    passed: bool
    n_required: int
    n_passed: int
    n_violated: int
    n_undecidable: int
    predicates: List[PredicateResult] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "hop_index": self.hop_index,
            "passed": self.passed,
            "n_required": self.n_required,
            "n_passed": self.n_passed,
            "n_violated": self.n_violated,
            "n_undecidable": self.n_undecidable,
            "predicates": [p.to_json() for p in self.predicates],
        }


def _label_value(facts: Mapping[str, Any], label: Optional[str]) -> Optional[Any]:
    """Read `entity_attrs[label]['value']` with hot-path scalar fallback."""
    if not label:
        return None
    # Hot-path scalar: parser promotes `score` / `highest_tile` / etc.
    if label in facts:
        v = facts[label]
        if v is not None:
            return v
    rec = (facts.get("entity_attrs") or {}).get(label)
    if not rec:
        return None
    if "value" in rec:
        return rec["value"]
    if "state" in rec:
        return rec["state"]
    return None


def _label_count(facts: Mapping[str, Any], label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    counts = facts.get("entity_label_count") or {}
    if label in counts:
        return int(counts[label])
    return None


def _as_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _entity_label_arg(predicate: Dict[str, Any]) -> Optional[str]:
    """The `args["entity_label"]` slot the lift mines from prose."""
    args = predicate.get("args") or {}
    raw = args.get("entity_label") or args.get("label")
    if not raw:
        return None
    return str(raw).strip()


def evaluate_predicate(
    predicate: Dict[str, Any],
    pre: StateSchema,
    post: StateSchema,
) -> PredicateResult:
    """Evaluate one effect predicate against the (pre, post) state pair.

    Returns a `PredicateResult`. `passed=None` means the predicate was
    *undecidable* (e.g. the labelled entity isn't present in either
    schema, or the metric isn't surfaced as a numeric value). Undecidable
    is treated as *non-blocking* by the default success scorer — it
    counts as "skip", not "fail".
    """

    ptype = str(predicate.get("type") or "").strip()
    pre_facts = pre.facts or {}
    post_facts = post.facts or {}

    if ptype == "cumulative_reward_increased":
        pre_score = _as_number(_label_value(pre_facts, "score"))
        post_score = _as_number(_label_value(post_facts, "score"))
        if pre_score is None or post_score is None:
            return PredicateResult(
                predicate=predicate, passed=None,
                detail="score not surfaced in facts",
            )
        ok = post_score > pre_score
        return PredicateResult(
            predicate=predicate, passed=ok,
            detail=f"score {pre_score!r} → {post_score!r}",
        )

    if ptype == "phase_transitioned":
        target = (predicate.get("args") or {}).get("to")
        pre_phase = pre_facts.get("phase")
        post_phase = post_facts.get("phase")
        if target is None and post_phase is not None:
            ok = post_phase != pre_phase
            return PredicateResult(
                predicate=predicate, passed=ok,
                detail=f"phase {pre_phase!r} → {post_phase!r}",
            )
        if target is None:
            return PredicateResult(
                predicate=predicate, passed=None,
                detail="no phase fact and no target arg",
            )
        ok = (post_phase == target) and (pre_phase != target)
        return PredicateResult(
            predicate=predicate, passed=ok,
            detail=f"phase {pre_phase!r} → {post_phase!r} (target={target!r})",
        )

    if ptype == "entity_value_increased":
        label = _entity_label_arg(predicate)
        pre_v = _as_number(_label_value(pre_facts, label))
        post_v = _as_number(_label_value(post_facts, label))
        if pre_v is None or post_v is None:
            return PredicateResult(
                predicate=predicate, passed=None,
                detail=f"entity {label!r} value not surfaced",
            )
        return PredicateResult(
            predicate=predicate, passed=post_v > pre_v,
            detail=f"{label}.value {pre_v!r} → {post_v!r}",
        )

    if ptype == "entity_value_decreased":
        label = _entity_label_arg(predicate)
        pre_v = _as_number(_label_value(pre_facts, label))
        post_v = _as_number(_label_value(post_facts, label))
        if pre_v is None or post_v is None:
            return PredicateResult(
                predicate=predicate, passed=None,
                detail=f"entity {label!r} value not surfaced",
            )
        return PredicateResult(
            predicate=predicate, passed=post_v < pre_v,
            detail=f"{label}.value {pre_v!r} → {post_v!r}",
        )

    if ptype == "entity_count_changed":
        label = _entity_label_arg(predicate)
        pre_c = _label_count(pre_facts, label)
        post_c = _label_count(post_facts, label)
        if pre_c is None or post_c is None:
            return PredicateResult(
                predicate=predicate, passed=None,
                detail=f"entity {label!r} count not surfaced",
            )
        return PredicateResult(
            predicate=predicate, passed=pre_c != post_c,
            detail=f"count[{label}] {pre_c} → {post_c}",
        )

    if ptype == "entity_appeared":
        label = _entity_label_arg(predicate)
        pre_c = _label_count(pre_facts, label) or 0
        post_c = _label_count(post_facts, label) or 0
        if label is None:
            return PredicateResult(predicate=predicate, passed=None,
                                   detail="entity_label arg missing")
        return PredicateResult(
            predicate=predicate, passed=post_c > pre_c,
            detail=f"count[{label}] {pre_c} → {post_c}",
        )

    if ptype == "entity_disappeared":
        label = _entity_label_arg(predicate)
        pre_c = _label_count(pre_facts, label) or 0
        post_c = _label_count(post_facts, label) or 0
        if label is None:
            return PredicateResult(predicate=predicate, passed=None,
                                   detail="entity_label arg missing")
        return PredicateResult(
            predicate=predicate, passed=post_c < pre_c,
            detail=f"count[{label}] {pre_c} → {post_c}",
        )

    if ptype == "attribute_changed":
        # Generic catch-all the lift uses for "X changes / shifts / moves".
        # We compare entity_attrs (label → field → value) shallowly. Any
        # difference in any tracked label counts as a change.
        pre_attrs = pre_facts.get("entity_attrs") or {}
        post_attrs = post_facts.get("entity_attrs") or {}
        if not pre_attrs and not post_attrs:
            return PredicateResult(
                predicate=predicate, passed=None,
                detail="entity_attrs missing on both sides",
            )
        # Restrict to labels present in both sides; new/missing labels are
        # entity_appeared/disappeared, not attribute_changed.
        shared = set(pre_attrs) & set(post_attrs)
        for lbl in shared:
            if pre_attrs[lbl] != post_attrs[lbl]:
                return PredicateResult(
                    predicate=predicate, passed=True,
                    detail=f"{lbl} attrs changed",
                )
        return PredicateResult(
            predicate=predicate, passed=False,
            detail="no shared label changed attributes",
        )

    return PredicateResult(
        predicate=predicate, passed=None,
        detail=f"unknown predicate type {ptype!r}",
    )


# ---------------------------------------------------------------------------
# Hop / episode roll-ups
# ---------------------------------------------------------------------------

def evaluate_hop_effects(
    hop: Dict[str, Any],
    pre: StateSchema,
    post: StateSchema,
    *,
    require_all: bool = True,
) -> HopEffectResult:
    """Evaluate every predicate in `hop["effects_add"]`.

    `require_all=True` (default): the hop passes iff every *decidable*
    predicate passes AND at least one decidable predicate exists. When
    every predicate is undecidable (e.g. cold-start prose for an unknown
    label), the hop result `passed=False` only if any predicate was
    explicitly violated; otherwise `passed=True` with `n_passed=0`. This
    keeps a hop without surface-able predicates from blocking transfer.
    """

    eff_add: List[Dict[str, Any]] = list(hop.get("effects_add") or [])
    results: List[PredicateResult] = [
        evaluate_predicate(p, pre, post) for p in eff_add
    ]
    n_required = len(eff_add)
    n_passed = sum(1 for r in results if r.passed is True)
    n_violated = sum(1 for r in results if r.passed is False)
    n_undecidable = sum(1 for r in results if r.passed is None)

    if n_violated > 0:
        passed = False
    elif require_all and n_required > 0:
        passed = n_violated == 0
    else:
        passed = True

    return HopEffectResult(
        hop_index=int(hop.get("hop_index", -1)),
        passed=passed,
        n_required=n_required,
        n_passed=n_passed,
        n_violated=n_violated,
        n_undecidable=n_undecidable,
        predicates=results,
    )


def evaluate_episode_effects(
    skill: Dict[str, Any] | Any,
    episode: SkillEpisode,
) -> Dict[str, Any]:
    """Per-step roll-up across an entire `SkillEpisode`.

    `skill` may be a `SkillRecord` or a plain dict — we only access the
    `protocol` list. For each `SkillEpisodeStep` that carries both
    `pre_state` and `post_state` snapshots, we evaluate that hop's
    `effects_add` predicates. Hops without snapshots are skipped (the
    adapter didn't record them, so we can't decide).

    Returns a JSON-serialisable dict with episode-level stats and a
    per-hop breakdown so callers can drill into which predicate failed.
    """

    protocol = getattr(skill, "protocol", None)
    if protocol is None and isinstance(skill, dict):
        protocol = skill.get("protocol")
    protocol = list(protocol or [])

    per_hop: List[Dict[str, Any]] = []
    n_total = 0
    n_pass = 0

    for i, step in enumerate(episode.steps):
        if i >= len(protocol):
            break
        hop = protocol[i] if isinstance(protocol[i], dict) else {}
        if not hop.get("effects_add"):
            continue
        if step.pre_state is None or step.post_state is None:
            continue
        pre = _hydrate_state(step.pre_state)
        post = _hydrate_state(step.post_state)
        res = evaluate_hop_effects({**hop, "hop_index": i}, pre, post)
        per_hop.append(res.to_json())
        n_total += 1
        if res.passed:
            n_pass += 1

    pass_rate = (n_pass / n_total) if n_total else 0.0
    return {
        "n_hops_evaluated": n_total,
        "n_hops_passed": n_pass,
        "pass_rate": pass_rate,
        "per_hop": per_hop,
    }


def _hydrate_state(snapshot: Dict[str, Any]) -> StateSchema:
    """Re-build a `StateSchema` from its serialized snapshot.

    `SkillEpisodeStep.pre_state` / `post_state` carry the dict form
    written by `StateSchema.to_json()`. We reconstruct only the fields
    the predicate evaluators read so we don't pay for the full round-trip
    (and don't have to deal with `EvidenceRef` reconstruction here).
    """

    return StateSchema(
        task=str(snapshot.get("task") or ""),
        domain=str(snapshot.get("domain") or "gymv"),
        elements=list(snapshot.get("elements") or []),
        facts=dict(snapshot.get("facts") or {}),
        inner_step=int(snapshot.get("inner_step") or 0),
        outer_step=int(snapshot.get("outer_step") or 0),
        extra=dict(snapshot.get("extra") or {}),
    )


# ---------------------------------------------------------------------------
# FewShotAdapter scorer
# ---------------------------------------------------------------------------

def make_per_step_success_fn(
    *,
    pass_rate_threshold: float = 1.0,
    require_episode_success: bool = True,
) -> Callable[[SkillEpisode, Any], float]:
    """Return a `SuccessFn` for `FewShotAdapter` that scores by per-hop
    effect-predicate pass rate.

    Args:
      pass_rate_threshold:
        The minimum fraction of *evaluated* hops that must pass for the
        shot to count as a success (`score == 1.0`). Default 1.0 — every
        evaluated hop must pass. Lower (e.g. 0.5) for noisy targets.
      require_episode_success:
        When True (default), the underlying `episode.outcome.success`
        must also be True. When False, only the predicate roll-up is
        consulted (used for the gate's "this skill *almost* worked"
        diagnostic).

    The returned function is intentionally hand-tuned to match the
    `FewShotAdapter`/`default_success_fn` signature, so the gate path
    remains a one-line opt-in:

        FewShotAdapter(harness=h, success_fn=make_per_step_success_fn())
    """

    def _score(episode: SkillEpisode, _demo: Any) -> float:
        if require_episode_success:
            out = episode.outcome
            if out is None or not out.success:
                return 0.0
        # Read the protocol from the episode's seed skill; we don't have
        # the raw `SkillRecord` here, but the harness writes per-hop
        # `pre_state`/`post_state` and the action_payload includes the
        # original effects_add roll-up via `extra`. Pull it from there.
        # Convention: `episode.outcome.extra["per_hop_effects"]` is set
        # by `GymvAdapter.run` when the executor records pre/post states.
        out = episode.outcome
        roll = (out.extra.get("per_hop_effects") if out and out.extra else None) or {}
        n_total = int(roll.get("n_hops_evaluated") or 0)
        n_pass = int(roll.get("n_hops_passed") or 0)
        if n_total == 0:
            # No predicates were evaluated (skill had no effects_add or
            # adapter didn't surface snapshots) — defer to outcome.success
            # so we don't penalise observational-only skills.
            return 1.0 if (out and out.success) else 0.0
        return 1.0 if (n_pass / n_total) >= pass_rate_threshold else 0.0

    return _score


# ---------------------------------------------------------------------------
# Day-6: domain-keyed success_fn registry.
#
# `make_per_step_success_fn` is the gymv-specific scorer that reads the
# per-hop `effects_add` predicates the lift mines and the GymvAdapter
# rolls into `episode.outcome.extra["per_hop_effects"]`. Other transfer
# targets (browser, osworld, video, visual_reasoning) need their *own*
# success_fns; the FewShotAdapter currently takes one fixed scorer at
# construction time.
#
# This registry lets the orchestrator say "give me the right scorer for
# `target_domain`" without wiring per-target callables manually:
#
#     success_fn = success_fn_for_domain("gymv")
#     adapter = FewShotAdapter(harness=h, success_fn=success_fn)
#
# Per-target scorers register via `register_success_fn(domain, factory)`
# at import time. Until they're written they fall back to
# `default_success_fn` (success ⇔ episode.outcome.success +
# contract_satisfied), so the lifecycle path stays well-defined.
# ---------------------------------------------------------------------------


# A "factory" is a zero-arg (or kwargs-only) callable that returns a
# concrete SuccessFn. Wrapping in a factory (rather than registering a
# bare SuccessFn) keeps callers from accidentally sharing per-skill
# state — the gymv scorer is stateless, but a future
# Stage-2 contract scorer might cache thresholds.
SuccessFnFactory = Callable[..., "Callable[[SkillEpisode, Any], float]"]


_DOMAIN_SUCCESS_FN_FACTORIES: Dict[str, SuccessFnFactory] = {}


def register_success_fn(domain: str, factory: SuccessFnFactory) -> None:
    """Register a domain-specific `SuccessFn` factory.

    Calling twice overwrites — explicitly intentional so test
    fixtures can swap a scorer without mutating module state.
    """
    _DOMAIN_SUCCESS_FN_FACTORIES[domain] = factory


def success_fn_for_domain(
    domain: str,
    *,
    pass_rate_threshold: float = 0.5,
    require_episode_success: bool = True,
    fallback: Optional["Callable[[SkillEpisode, Any], float]"] = None,
) -> "Callable[[SkillEpisode, Any], float]":
    """Look up the registered scorer for ``domain`` and instantiate it.

    Returns ``fallback`` (default: `default_success_fn` from
    `harness.few_shot_adapter`) when no scorer is registered. The
    factory is called with the standard kwargs every gymv-style scorer
    accepts (`pass_rate_threshold`, `require_episode_success`); a
    domain-specific factory that doesn't take those simply ignores
    them.
    """
    factory = _DOMAIN_SUCCESS_FN_FACTORIES.get(domain)
    if factory is None:
        if fallback is not None:
            return fallback
        # Late import to avoid the harness/init cycle (few_shot_adapter
        # imports from gymv_success indirectly via __init__).
        from harness.few_shot_adapter import default_success_fn
        return default_success_fn
    return factory(
        pass_rate_threshold=pass_rate_threshold,
        require_episode_success=require_episode_success,
    )


def registered_success_fn_domains() -> Tuple[str, ...]:
    """Sorted view of currently-registered domains (mostly for tests
    and the diagnostic banner)."""
    return tuple(sorted(_DOMAIN_SUCCESS_FN_FACTORIES.keys()))


# Bootstrap: gymv is always registered (it's the source domain).
register_success_fn("gymv", make_per_step_success_fn)


__all__ = [
    "EFFECT_PREDICATE_TYPES",
    "HopEffectResult",
    "PredicateResult",
    "SuccessFnFactory",
    "evaluate_episode_effects",
    "evaluate_hop_effects",
    "evaluate_predicate",
    "make_per_step_success_fn",
    "register_success_fn",
    "registered_success_fn_domains",
    "success_fn_for_domain",
]
