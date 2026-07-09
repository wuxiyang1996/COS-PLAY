"""Build ``FewShotDemo``s from cold-start OSWorld episodes.

Stage 3 (rollout memo §6.1, §11.5.5) lifts the cold-start OSWorld
corpus (``Cold-start-out-osworld/<ts>/<domain>/<task-uuid>/episode_*.json``)
into per-step ``FewShotDemo``s. Layout differs from the gymv
counterpart (:mod:`harness.few_shot_demos_gymv`): three nested levels
under the cold-start root (timestamp dir → domain dir → task-uuid
dir), and we read the ``metadata.schema_canonical`` AT-SPI heuristic
head (NOT ``metadata.schema``, which is the VLM head with
hallucinations). ``state.task`` is retagged to the friendly domain
name (``"vlc"``) so the eligibility filter sees the right task;
``state.domain`` becomes ``"osworld"``.

Bindings: SoM actions like ``click_element(id=N)`` populate
``target`` / ``som_id``; raw pyautogui calls (``pyautogui.click(x,
y)``) populate ``x`` / ``y``. ``WAIT`` / ``DONE`` get empty
bindings so the executor's catch-all branch fires.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.few_shot_adapter import FewShotDemo
from labeling_supplement._harness_io_helpers import parse_schema_canonical

logger = logging.getLogger("harness.few_shot_demos_osworld")


__all__ = [
    "build_demos_from_osworld_episodes",
    "build_demos_from_osworld_episode_file",
]


_RX_CLICK_ELEMENT = re.compile(
    r"click_element\s*\(\s*id\s*=\s*(\d+)", re.IGNORECASE,
)
_RX_PYAUTOGUI_XY = re.compile(
    r"pyautogui\.\w+\s*\(\s*(-?\d+)\s*,\s*(-?\d+)", re.IGNORECASE,
)


def _bindings_for_action(action: str) -> Dict[str, Any]:
    """Return slot bindings for one OSWorld action string.

    Recognised shapes: SoM ``click_element(id=N)`` (populate
    ``target`` / ``som_id``), raw ``pyautogui.click(x, y)`` /
    ``pyautogui.doubleClick(x, y)`` (populate ``x`` / ``y``), and
    ``WAIT`` / ``DONE`` (empty). Unrecognised actions fall through to
    the executor's catch-all.
    """

    if not isinstance(action, str) or not action.strip():
        return {}
    a = action.strip()
    upper = a.upper()
    if upper in {"WAIT", "DONE", "FINISH"}:
        return {}

    m = _RX_CLICK_ELEMENT.search(a)
    if m:
        som_id = m.group(1)
        return {"target": som_id, "som_id": som_id}

    m = _RX_PYAUTOGUI_XY.search(a)
    if m:
        return {"x": m.group(1), "y": m.group(2)}

    return {}


def build_demos_from_osworld_episode_file(
    ep_path: Path,
    *,
    domain: str,
    domain_tag: str = "osworld",
    max_demos: int = 2,
    skip_noop: bool = True,
) -> List[FewShotDemo]:
    """Load one ``episode_*.json`` and emit up to ``max_demos`` demos.

    Robust to missing fields — a step without a parseable
    ``schema_canonical`` is silently skipped (logged at DEBUG).
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
            logger.debug(
                "step %s in %s has no schema_canonical block",
                step.get("step_id"), ep_path.name,
            )
            continue
        try:
            state = parse_schema_canonical(sc, default_domain=domain_tag)
        except Exception as exc:                                        # noqa: BLE001
            logger.debug(
                "parse_schema_canonical failed (%r) for %s",
                exc, ep_path.name,
            )
            continue

        action = step.get("action") or step.get("action_taken") or ""
        if skip_noop and isinstance(action, str) and action.strip().upper() in {"WAIT"}:
            # WAIT steps don't tell us anything about the skill's
            # effect on the desktop — drop unless the caller opted in.
            continue
        if isinstance(action, str) and action.strip().upper() == "DONE":
            # DONE is the actor's "I'm finished" sentinel. Skip when
            # noop-skipping is on; the demo before this one already
            # captures the meaningful state.
            if skip_noop:
                continue

        # Retag the state so the eligibility filter and adapter
        # dispatch see the OSWorld domain + the cold-start task name
        # (the friendly domain like "vlc", not the task UUID).
        interface = step.get("interface") or {}
        cs_domain = interface.get("domain") or domain
        cs_task_id = interface.get("task_id") or data.get("game_name") or ""

        state.task = str(cs_domain)
        state.domain = domain_tag

        bindings = _bindings_for_action(action) if isinstance(action, str) else {}
        reward = float(step.get("reward") or 0.0)
        expected: Dict[str, Any] = {
            "reward": reward,
            "action": action,
            "step_id": step.get("step_id"),
            "episode": ep_path.stem,
            "domain": str(cs_domain),
            "task_uuid": str(cs_task_id),
        }
        out.append(
            FewShotDemo(
                state=state,
                bindings=bindings,
                expected=expected,
                notes=f"cold_start_osworld:{ep_path.parent.name}:{ep_path.stem}:step={step.get('step_id')}",
            )
        )
    return out


def build_demos_from_osworld_episodes(
    cold_start_root: Path,
    *,
    domain: str,
    max_episodes: int = 3,
    max_demos_per_episode: int = 2,
    domain_tag: str = "osworld",
    skip_noop: bool = True,
) -> List[FewShotDemo]:
    """Walk ``cold_start_root/*/<domain>/*/episode_*.json`` and emit
    up to ``max_episodes × max_demos_per_episode`` demos.

    ``cold_start_root``'s immediate children are timestamped run dirs
    (``2026-05-01_06-54-07``); ``domain`` is the OSWorld domain name
    (``"vlc"``, ``"vs_code"``, …) and becomes ``state.task`` on every
    demo. ``domain_tag`` (default ``"osworld"``) becomes
    ``state.domain``. ``skip_noop=True`` drops ``is_noop`` / ``WAIT``
    / ``DONE`` steps. Returns demos in deterministic sort order.
    """

    if not cold_start_root.exists():
        logger.warning("cold_start_root missing: %s", cold_start_root)
        return []

    out: List[FewShotDemo] = []
    n_files_seen = 0
    # Layout: <cold_start_root>/<timestamp>/<domain>/<task-uuid>/episode_*.json
    candidates = sorted(cold_start_root.glob(f"*/{domain}/*/episode_*.json"))
    for ep_path in candidates:
        if n_files_seen >= max_episodes:
            break
        # Skip files that don't sit under a real timestamped run dir
        # (e.g. ``latest`` symlinks pointing back to a ``2026-…`` dir
        # produce duplicates that aren't worth re-processing).
        ts_dir = ep_path.parent.parent.parent
        if ts_dir.is_symlink():
            logger.debug("skipping symlinked run dir: %s", ts_dir)
            continue
        n_files_seen += 1
        ep_demos = build_demos_from_osworld_episode_file(
            ep_path,
            domain=domain,
            domain_tag=domain_tag,
            max_demos=max_demos_per_episode,
            skip_noop=skip_noop,
        )
        out.extend(ep_demos)
    logger.info(
        "built %d osworld demo(s) for domain=%s from %d episode file(s) "
        "(under %s)",
        len(out), domain, n_files_seen, cold_start_root,
    )
    return out
