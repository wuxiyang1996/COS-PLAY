"""Build `FewShotDemo`s from cold-start gymv episodes.

PLAN-HARNESS §22 Day-5: Stage 3a `FewShotAdapter` cross-task transfer
needs **target-task demos** to feed `adapt(skill, target_domain=…,
demos=…, target_task=…)`. The cold-start corpus
(`labeling/skill_actions_out/run_<ts>/env_wrappers/<game>/episode_*.json`)
already records every step a VLM took on each game, with:

  * ``metadata.schema_canonical`` — the canonical ``<state>...</state>``
    block at that step (Day-3 / Day-4B format),
  * ``action`` — the env-side action token (``up`` / ``left`` / …),
  * ``reward`` — the per-step reward,
  * ``intentions`` — the lifted intent tags.

This module wraps that corpus into ``FewShotDemo``s so a 2048-feasible
skill can run against tetris demos (or vice versa) end-to-end through
``harness.run_skill``, with a domain-aware success_fn that scores via
the Day-3 / Day-4B effect-predicate machinery.

Public API:

    demos = build_demos_from_episodes(
        actions_root, corpus="env_wrappers", game="tetris",
        max_episodes=3, max_demos_per_episode=2,
    )

A demo's `state` is a fully-parsed `StateSchema` (Day-3 helpers); the
`bindings` carry the action the cold-start agent took at that step
(so a SLIDE.direction=… or MOVE.direction=… hop has a concrete
binding to play through), and `expected` records the per-step reward
for downstream sanity checks.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common.state_schema import StateSchema
from harness.few_shot_adapter import FewShotDemo
from labeling_supplement._harness_io_helpers import parse_schema_canonical

logger = logging.getLogger("harness.few_shot_demos_gymv")


__all__ = [
    "build_demos_from_episodes",
    "build_demos_from_episode_file",
]


def build_demos_from_episodes(
    actions_root: Path,
    *,
    corpus: str = "env_wrappers",
    game: str,
    max_episodes: int = 3,
    max_demos_per_episode: int = 2,
    domain: str = "gymv",
    skip_noop: bool = True,
) -> List[FewShotDemo]:
    """Walk ``actions_root/<corpus>/<game>/episode_*.json`` and harvest
    up to ``max_episodes × max_demos_per_episode`` demos. Each demo is
    one ``(schema_canonical, action, reward)`` triplet from a cold-start
    rollout.

    ``skip_noop=True`` drops steps where ``metadata.is_noop`` is true —
    those steps don't tell us anything about the skill's effect on the
    env. Demos preserve the cold-start episode order; callers wanting
    randomness should shuffle the result themselves.
    """
    src = actions_root / corpus / game
    if not src.exists():
        logger.warning("demo source missing: %s", src)
        return []
    out: List[FewShotDemo] = []
    for ep_path in sorted(src.glob("episode_*.json"))[:max_episodes]:
        ep_demos = build_demos_from_episode_file(
            ep_path,
            game=game,
            domain=domain,
            max_demos=max_demos_per_episode,
            skip_noop=skip_noop,
        )
        out.extend(ep_demos)
    logger.info(
        "built %d demo(s) for game=%s from %d episode file(s)",
        len(out), game, min(max_episodes, len(list(src.glob("episode_*.json")))),
    )
    return out


def build_demos_from_episode_file(
    ep_path: Path,
    *,
    game: str,
    domain: str = "gymv",
    max_demos: int = 2,
    skip_noop: bool = True,
) -> List[FewShotDemo]:
    """Load a single ``episode_NNN.json`` file and emit up to
    ``max_demos`` `FewShotDemo`s.

    Robust to missing fields — a step without a parseable
    `schema_canonical` is silently skipped (logged at DEBUG).
    """
    try:
        data = json.loads(ep_path.read_text())
    except Exception as exc:                                            # noqa: BLE001
        logger.warning("failed to read %s: %r", ep_path, exc)
        return []

    out: List[FewShotDemo] = []
    experiences = data.get("experiences") or []
    for step in experiences:
        if len(out) >= max_demos:
            break
        md = step.get("metadata") or {}
        if skip_noop and bool(md.get("is_noop", False)):
            continue
        sc = md.get("schema_canonical") or ""
        if "<state>" not in sc:
            logger.debug("step %s in %s has no schema_canonical block",
                         step.get("step_id"), ep_path.name)
            continue
        try:
            state = parse_schema_canonical(sc, default_domain=domain)
        except Exception as exc:                                        # noqa: BLE001
            logger.debug("parse_schema_canonical failed (%r) for %s",
                         exc, ep_path.name)
            continue
        # Tag the state with the target task so the eligibility filter
        # and adapter dispatch see what the FewShotAdapter is probing.
        state.task = state.task or f"make_gaming_env/{game}"
        action = step.get("action") or step.get("action_taken") or ""
        bindings: Dict[str, Any] = {}
        if isinstance(action, str) and action:
            # The cold-start prose's protocols mostly carry SLIDE /
            # MOVE / SELECT hops keyed on `direction` or `target`;
            # populate both so the executor's payload-value rescue
            # clause can resolve.
            bindings["direction"] = action
            bindings["target"] = action
        reward = float(step.get("reward") or 0.0)
        expected: Dict[str, Any] = {
            "reward": reward,
            "action": action,
            "step_id": step.get("step_id"),
            "episode": ep_path.stem,
        }
        out.append(
            FewShotDemo(
                state=state,
                bindings=bindings,
                expected=expected,
                notes=f"cold_start:{ep_path.stem}:step={step.get('step_id')}",
            )
        )
    return out
