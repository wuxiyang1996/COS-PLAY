"""OSWorld per-step success function — predicate evaluation against
pre/post ``StateSchema`` snapshots.

Stage 3 (rollout memo §6.1, §11.5.5) builds OSWorld facts via the
:mod:`harness.osworld_schema_producer`, which emits the same
``entity_attrs`` / ``entity_label_count`` / ``state_flags.{phase,
progress, error}`` shape gymv uses. The lifted protocol's effect
predicates (``entity_value_increased``, ``cumulative_reward_increased``,
``phase_transitioned``, …) are therefore evaluable by the existing
gymv evaluator with no domain-specific branches.

This module:

  1. Re-exports ``make_osworld_per_step_success_fn`` — a thin wrapper
     around ``harness.gymv_success.make_per_step_success_fn`` so the
     orchestrator's ``success_fn_for_domain('osworld')`` lookup
     returns a working scorer.
  2. Registers itself with ``register_success_fn('osworld', …)`` at
     import time, mirroring the gymv bootstrap.

Future cuts (deferred to Stage-3-second-cut, §11.5.5 closing notes)
can override OSWorld-specific predicate types — e.g. ``task_status``,
``last_action``, ``actor_used_action``, ``visited_entity`` — by
re-routing those predicate types to a custom evaluator before
delegating to the gymv path. The first cut delegates wholesale.
"""

from __future__ import annotations

from typing import Any, Callable

from data_structure.extensions.skill_episode import SkillEpisode
from harness.gymv_success import (
    make_per_step_success_fn as _make_gymv_success,
    register_success_fn,
)


__all__ = ["make_osworld_per_step_success_fn"]


def make_osworld_per_step_success_fn(
    *,
    pass_rate_threshold: float = 1.0,
    require_episode_success: bool = True,
) -> Callable[[SkillEpisode, Any], float]:
    """Per-hop effect-predicate success_fn for OSWorld.

    Delegates to ``harness.gymv_success.make_per_step_success_fn``
    because the OSWorld producer emits gymv-shape facts:
    ``entity_attrs`` (label → field → value), ``entity_label_count``
    (label → count), and ``state_flags.{phase, progress, error}``. The
    lift's effect-predicate vocabulary
    (``entity_value_increased``, ``entity_count_changed``,
    ``cumulative_reward_increased``, ``phase_transitioned``, …) is the
    same on both sides, so the same evaluator applies.

    Future cuts can override OSWorld-specific predicate types
    (``task_status``, ``last_action``, ``actor_used_action``,
    ``visited_entity``) by routing those types to a custom branch
    before falling through to gymv. Stage 3's first cut keeps it
    simple and unblocks the dispatcher.

    Args:
      pass_rate_threshold: Minimum fraction of *evaluated* hops that
        must pass for the shot to count as a success. Default 1.0 —
        every evaluated hop must pass. Lower (e.g. 0.5) for noisy
        targets.
      require_episode_success: When True (default), the underlying
        ``episode.outcome.success`` must also be True. When False,
        only the predicate roll-up is consulted.

    Returns: a ``SuccessFn`` (``Callable[[SkillEpisode, Any], float]``)
    suitable for ``FewShotAdapter(success_fn=…)``.
    """

    return _make_gymv_success(
        pass_rate_threshold=pass_rate_threshold,
        require_episode_success=require_episode_success,
    )


# Register at import time so ``success_fn_for_domain('osworld')``
# resolves without having to import this module separately. The
# registry write is idempotent — re-importing is a no-op.
register_success_fn("osworld", make_osworld_per_step_success_fn)
