"""Build `FewShotDemo`s from cold-start BrowserGym episodes.

Stage 4 of the cross-domain transfer rollout (memo §6.1, §11.5.5).
Mirrors `harness/few_shot_demos_gymv.py` but reads from BrowserGym's
cold-start corpus: ``Cold-start-out-browsergym/<task_id>/episode_*.json``
with ``<task_id>`` = ``<prefix>.<rest>`` (e.g. ``assistantbench.test.92``).

Each non-noop step becomes a `FewShotDemo` whose `state` is parsed
from `metadata.schema_canonical` (re-tagged to ``domain="browser"``),
`bindings` carries the parsed action's args (e.g. ``click("e20")``
→ ``{"target": "e20", "bid": "e20"}``), and `expected` records
reward / action / URL / focused-bid for downstream checks.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from harness.few_shot_adapter import FewShotDemo
from labeling_supplement._harness_io_helpers import parse_schema_canonical

logger = logging.getLogger("harness.few_shot_demos_browsergym")


__all__ = [
    "build_demos_from_browsergym_episodes",
    "build_demos_from_browsergym_episode_file",
    "parse_browsergym_action",
]


# `click("e20")`, `fill("BID", "text")`, `scroll(0, 300)`, `noop()`,
# `key_press("Enter")`, `select_option("BID", "value")`, `goto("url")`
# … the BrowserGym highlevel parser allows quoted string args and
# bare numeric args. Defensive against single/double quotes,
# trailing semicolons, etc.
_ACTION_RX = re.compile(
    r"\s*(?P<verb>[a-zA-Z_][\w]*)\s*\((?P<args>.*?)\)\s*$",
    re.DOTALL,
)
_ARG_RX = re.compile(
    r"""\s*(?:"(?P<dq>(?:\\.|[^"\\])*)"|'(?P<sq>(?:\\.|[^'\\])*)'"""
    r"""|(?P<bare>[^,]+?))\s*(?:,|$)""",
    re.DOTALL,
)


def parse_browsergym_action(action: str) -> tuple[str, List[str]]:
    """Best-effort split of ``click("e20")`` → ``("click", ["e20"])``.

    Returns ``("", [])`` for unparseable inputs. Quoted args are
    returned without their quotes; bare args are trimmed.
    """
    if not action or not isinstance(action, str):
        return "", []
    m = _ACTION_RX.match(action.strip().rstrip(";"))
    if not m:
        return "", []
    verb = m.group("verb").lower()
    args_blob = m.group("args") or ""
    args: List[str] = []
    if args_blob.strip():
        for a in _ARG_RX.finditer(args_blob):
            if a.group("dq") is not None:
                args.append(a.group("dq"))
            elif a.group("sq") is not None:
                args.append(a.group("sq"))
            elif a.group("bare") is not None:
                args.append(a.group("bare").strip())
    return verb, args


def _bindings_from_action(verb: str, args: List[str]) -> Dict[str, Any]:
    """Map ``(verb, args)`` to a payload-bindings dict the adapter's
    slot resolver can consume. Conventions:

      * ``click(BID)`` / ``check`` / ``uncheck`` / ``hover`` →
        ``{"target": BID, "bid": BID}``.
      * ``fill(BID, text)`` → ``{"target": BID, "bid": BID, "text": text}``.
      * ``select_option(BID, value)`` → ``{"target": BID, "bid": BID, "value": value}``.
      * ``scroll(dx, dy)`` → ``{"dx": dx, "dy": dy}``.
      * ``key_press(key)`` → ``{"key": key}``.
      * ``goto(url)`` → ``{"url": url}``.
      * ``noop()`` / ``go_back()`` / etc. → ``{}``.
    """
    if not verb:
        return {}
    if verb in {"click", "check", "uncheck", "hover"} and args:
        return {"target": args[0], "bid": args[0]}
    if verb == "fill" and len(args) >= 2:
        return {"target": args[0], "bid": args[0], "text": args[1]}
    if verb in {"select_option", "select"} and len(args) >= 2:
        return {"target": args[0], "bid": args[0], "value": args[1]}
    if verb == "scroll" and len(args) >= 2:
        return {"dx": args[0], "dy": args[1]}
    if verb in {"key_press", "press", "keyboard_press"} and args:
        return {"key": args[0]}
    if verb == "goto" and args:
        return {"url": args[0]}
    return {}


def build_demos_from_browsergym_episodes(
    cold_start_root: Path,
    *,
    task_prefix: str,
    max_episodes: int = 3,
    max_demos_per_episode: int = 2,
    domain_tag: str = "browser",
    skip_noop: bool = True,
) -> List[FewShotDemo]:
    """Walk ``cold_start_root/<task_prefix>.*/episode_*.json`` and
    harvest up to ``max_episodes × max_demos_per_episode`` demos.

    The ``.`` separator after ``task_prefix`` is intentional —
    BrowserGym task IDs are ``<prefix>.<rest>``. Steps where
    ``metadata.is_noop`` is True are skipped when ``skip_noop=True``.
    """
    if not cold_start_root.exists():
        logger.warning("cold_start_root missing: %s", cold_start_root)
        return []
    pattern = f"{task_prefix}.*/episode_*.json"
    episodes = sorted(cold_start_root.glob(pattern))[:max_episodes]
    out: List[FewShotDemo] = []
    for ep_path in episodes:
        out.extend(build_demos_from_browsergym_episode_file(
            ep_path,
            domain_tag=domain_tag,
            max_demos=max_demos_per_episode,
            skip_noop=skip_noop,
        ))
    logger.info(
        "built %d demo(s) for task_prefix=%s from %d episode file(s)",
        len(out), task_prefix, len(episodes),
    )
    return out


def build_demos_from_browsergym_episode_file(
    ep_path: Path,
    *,
    domain_tag: str = "browser",
    max_demos: int = 2,
    skip_noop: bool = True,
) -> List[FewShotDemo]:
    """Load a single ``episode_NNN.json`` and emit up to
    ``max_demos`` `FewShotDemo`s. Steps without a parseable
    ``schema_canonical`` are silently skipped.
    """
    try:
        data = json.loads(ep_path.read_text())
    except Exception as exc:                                            # noqa: BLE001
        logger.warning("failed to read %s: %r", ep_path, exc)
        return []

    out: List[FewShotDemo] = []
    experiences = data.get("experiences") or []
    interface = data.get("interface") or {}
    game_name = (
        data.get("game_name")
        or interface.get("game_name")
        or ep_path.parent.name
    )
    for step in experiences:
        if len(out) >= max_demos:
            break
        md = step.get("metadata") or {}
        if skip_noop and bool(md.get("is_noop", False)):
            continue
        sc = md.get("schema_canonical") or ""
        if "<state>" not in sc:
            continue
        try:
            state = parse_schema_canonical(sc, default_domain=domain_tag)
        except Exception:                                               # noqa: BLE001
            continue
        # Re-tag defensively — cold-start corpus is ``domain=browser``
        # but a future re-shard could land with a stale tag.
        state.domain = domain_tag
        step_iface = step.get("interface") or interface
        state.task = state.task or step_iface.get("game_name") or game_name

        action_str = step.get("action") or step.get("action_taken") or ""
        verb, args = parse_browsergym_action(str(action_str))
        bindings = _bindings_from_action(verb, args)

        expected: Dict[str, Any] = {
            "reward": float(step.get("reward") or 0.0),
            "action": action_str,
            "step_id": step.get("step_id"),
            "episode": ep_path.stem,
            "game_name": step_iface.get("game_name") or game_name,
            "url": md.get("url"),
            "focused_bid": md.get("focused_element_bid"),
        }
        out.append(
            FewShotDemo(
                state=state,
                bindings=bindings,
                expected=expected,
                notes=(
                    f"cold_start_browsergym:{ep_path.parent.name}:"
                    f"step={step.get('step_id')}"
                ),
            )
        )
    return out
