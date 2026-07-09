"""BrowserGym per-step success_fn factory.

PLAN-HARNESS §22 / cross-domain transfer suite rollout memo §11.5.5.
This is the per-domain registry entry for ``"browser"`` so the
``FewShotAdapter`` can pick up the right scorer for BrowserGym
transfer cells via ``success_fn_for_domain("browser", ...)``.

Stage 4's first cut delegates to gymv's per-hop effect-predicate
evaluator (``harness.gymv_success.make_per_step_success_fn``) since
the BrowserGym schema producer emits gymv-shape facts:

  * ``state_flags.phase`` / ``state_flags.progress`` /
    ``state_flags.error`` / ``state_flags.dialog_open`` —
    ``phase_transitioned`` / ``cumulative_reward_increased`` baseline.
  * ``entity_attrs`` (label → field → value) — ``attribute_changed``
    predicate input.
  * ``entity_label_count`` — ``entity_count_changed`` /
    ``entity_appeared`` / ``entity_disappeared`` predicate input.

Future cuts can override the browser-specific predicates
(``url_changed``, ``page_loaded``, ``form_submitted``,
``modal_dismissed``) by intercepting the predicate evaluator before
delegating; until those land, the gymv evaluator's catch-all
``attribute_changed`` covers BrowserGym DOM mutations adequately.

Reference:
  * ``harness/gymv_success.py`` — the canonical per-hop evaluator.
  * ``harness/qa_success.py`` — the visual_reasoning analogue.
  * ``harness/__init__.py`` — registration site.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from data_structure.extensions.skill_episode import SkillEpisode
from harness.gymv_success import make_per_step_success_fn as _make_gymv_success
from harness.gymv_success import register_success_fn

logger = logging.getLogger("harness.browser_success")


__all__ = ["make_browser_per_step_success_fn"]


def make_browser_per_step_success_fn(
    *,
    pass_rate_threshold: float = 1.0,
    require_episode_success: bool = True,
) -> Callable[[SkillEpisode, Any], float]:
    """Per-hop effect-predicate success_fn for BrowserGym.

    Delegates to gymv's evaluator since the producer emits gymv-shape
    facts (entity_attrs, entity_label_count, state_flags.{phase,
    progress, error, dialog_open}). Future cuts can override the
    browser-specific predicates (url_changed, page_loaded,
    form_submitted, modal_dismissed) — see the module docstring.

    Kwargs match the registry contract (gymv_success.py:540-543) so
    ``success_fn_for_domain("browser", pass_rate_threshold=...,
    require_episode_success=...)`` resolves uniformly.
    """
    return _make_gymv_success(
        pass_rate_threshold=pass_rate_threshold,
        require_episode_success=require_episode_success,
    )


# Bootstrap: register the factory at import time so callers that
# reach `from harness.gymv_success import success_fn_for_domain`
# after `import harness.browser_success` see the entry. The
# top-level `harness/__init__.py` also imports this module so the
# default `import harness` triggers registration.
register_success_fn("browser", make_browser_per_step_success_fn)
