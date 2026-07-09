"""Command-line entrypoint for SFT data collection with GPT-4o.

Drives :class:`~decision_agents.SFT.actor_gpt4o.GPT4oCollectorActor`
against an environment factory, dumping per-step rows the existing
``trainer/SFT`` pipeline can train on without modification.

Example
-------
::

    python -m decision_agents.SFT.run_collect \\
        --game tetris \\
        --episodes 50 \\
        --max-steps 200 \\
        --out labeling/output/gpt54_skill_labeled/grpo_coldstart \\
        --env-factory my_pkg.envs:make_tetris \\
        --schema-from-info schema_text

The ``--env-factory`` argument is a ``module:callable`` reference that
returns a Gym-like env (``reset()``, ``step(action)``).  Anything that
returns ``(obs, info)`` from reset and ``(obs, reward, term, trunc, info)``
from step works.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from decision_agents.SFT.actor_gpt4o import GPT4oCollectorActor
from decision_agents.SFT.sft_recorder import (
    DEFAULT_SFT_OUTPUT_DIR,
    SFTRecorder,
)
from decision_agents.core.multimodal import VisualInput

_LOGGER = logging.getLogger(__name__)


def _load_factory(spec: str) -> Callable[..., Any]:
    """Resolve a ``"module:callable"`` reference."""
    if ":" not in spec:
        raise ValueError(f"--env-factory must be 'module:callable', got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, attr, None)
    if not callable(factory):
        raise ValueError(f"{spec!r} is not a callable")
    return factory


def _image_from_info(info: Dict[str, Any], key: str) -> Optional[VisualInput]:
    """Extract a :class:`VisualInput` from ``info[key]``.

    Accepts a path string, a ``VisualInput`` instance, or a dict with
    keys matching :class:`VisualInput` fields.  Returns ``None`` when
    the field is missing — useful for environments that only expose
    screenshots on a subset of steps.
    """
    raw = (info or {}).get(key)
    if raw is None:
        return None
    if isinstance(raw, VisualInput):
        return raw
    if isinstance(raw, str):
        return VisualInput(image_path=raw)
    if isinstance(raw, dict):
        return VisualInput(**raw)
    _LOGGER.debug("Unknown image payload type %r for key %s", type(raw), key)
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="decision_agents.SFT.run_collect",
        description="Collect SFT cold-start data with GPT-4o.",
    )
    p.add_argument("--env-factory", required=True,
                   help="'module:callable' returning a Gym-like env")
    p.add_argument("--game", required=True, help="Game / domain id (e.g. tetris)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--task", default="", help="Optional task string for the actor")
    p.add_argument("--out", type=Path, default=DEFAULT_SFT_OUTPUT_DIR,
                   help="Output dir; <out>/<game>/{skill_selection,action_taking}.jsonl")
    p.add_argument("--model", default="gpt-4o",
                   help="GPT model id (only gpt-4o has been quality-checked)")
    p.add_argument(
        "--image-info-key",
        default="screenshot",
        help="info-dict key the env uses for the per-step image "
             "(value can be a path string, dict, or VisualInput).",
    )
    p.add_argument(
        "--schema-info-key",
        default="schema_text",
        help="info-dict key for the parsed <state> schema text",
    )
    p.add_argument("--no-vision", action="store_true",
                   help="Disable image attachment even if the env exposes one "
                        "(useful for text-only debugging)")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    factory = _load_factory(args.env_factory)
    recorder = SFTRecorder(output_dir=args.out)
    actor = GPT4oCollectorActor(
        recorder=recorder,
        game=args.game,
        model=args.model,
    )

    total_steps = 0
    for ep in range(args.episodes):
        env = factory()
        actor.reset()
        obs, info = env.reset()
        observation = str(obs) if obs is not None else ""
        info = dict(info or {})
        done = False
        step_count = 0
        while step_count < args.max_steps and not done:
            schema_text = info.get(args.schema_info_key)
            valid_actions = info.get("valid_actions") or info.get("available_actions")
            image = None if args.no_vision else _image_from_info(info, args.image_info_key)
            decision = actor.step(
                observation=observation,
                schema_text=schema_text,
                task=args.task or info.get("task", ""),
                valid_actions=valid_actions,
                info=info,
                images=[image] if image is not None else None,
            )
            env_action = (
                decision.resolved.resolved
                if decision.resolved is not None
                else decision.action
            )
            next_obs, reward, term, trunc, next_info = env.step(env_action)
            done = bool(term or trunc)
            actor.observe_result(
                decision,
                reward=float(reward or 0.0),
                next_observation=str(next_obs) if next_obs is not None else "",
                next_schema_text=(next_info or {}).get(args.schema_info_key),
                done=done,
            )
            observation = str(next_obs) if next_obs is not None else ""
            info = dict(next_info or {})
            step_count += 1
        total_steps += step_count
        _LOGGER.info("episode %d/%d: %d steps", ep + 1, args.episodes, step_count)

    manifest = recorder.write_manifest()
    print(json.dumps({
        "manifest": str(manifest),
        "total_steps": total_steps,
        "stats": recorder.stats(),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
