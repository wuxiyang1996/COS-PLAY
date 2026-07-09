"""Deterministic dict-→-``<state>`` markup unifier for cross-domain schema.

Goal
----
Every domain we train/eval on (gymv, gamingagent, osworld, browsergym, orak)
ships its own ``info["structured_state"]`` shape — when it ships one at all.
This module produces a *single* canonical ``<state>...</state>`` markup
string from any of those shapes so that:

* the actor (or any downstream consumer) sees byte-identical schema at
  training and evaluation time for the same observation,
* the Crafter / Harness / GameProfile generators have one unified vocabulary
  (entities / attributes / state_flags / targets / actions) regardless of
  which env produced the trajectory,
* the markup conforms to the ontology vocabulary defined in
  :mod:`vlm_wrapper.schema` so existing parsers (`parse_schema_output`,
  `validate_schema`) accept it without modification.

The renderers are **pure-Python, deterministic, zero-LLM**.  They always
return a valid markup; on any unexpected input they fall back to a minimal
``e1=environment / e2=observation`` form so the caller never has to wrap
this in try/except.

Per-domain coverage today
-------------------------
* ``gymv_temporal``:  rich (display_name, genre, ram_watch, parsed_text,
                         entities, available_actions)
* ``gamingagent``:    rich (board, max_tile, empty, merges, phase,
                         affordance) — 2048 / candy_crush / tetris
* ``osworld``:        moderate (a11y-derived screen_regions,
                         ui_element_count, terminal, has_dialog)
* ``browsergym``:     defers to
                         :func:`browsergym_wrapper.heuristic.obs_to_schema`
                         which already emits canonical markup
* ``orak``:           **degraded** — no structured head exists today, so
                         we emit a minimal markup carrying obs_nl + score.
                         A proper structured head per Orak game is the
                         next milestone.

Phase 1 only exercises the gymv path during training; the rest are
covered for forward compatibility (eval time, future curricula).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Aligned with :data:`vlm_wrapper.schema.SCHEMA_VERSION` so that the
# deterministic-head markup and the VLM-head markup both stamp the same
# contract version.  Bump in lock-step with that constant whenever the
# section / vocabulary contracts change.  Downstream cache keys
# (e.g. game_schema artifacts) include this so a vocabulary bump
# invalidates stale caches.
SCHEMA_VERSION = "0.2"

DEFAULT_MAX_ENTITIES = 12

# ── Ontology → affordance verbs ───────────────────────────────────────
# Cross-domain fallback affordance lists, derived from the cold-start
# few-shot gold examples in ``vlm_wrapper/few_shot_examples/*.txt``.
# Two variants per ontology: a ``game`` flavour (track / select-centric)
# and a ``desktop`` flavour (focus / open-centric).  ``textual_anchor``
# and ``goal_indicator`` deliberately have **no** affordance row in the
# gold examples — we replicate that omission here.

_AFFORD_GAME: Dict[str, List[str]] = {
    "selectable_entity":  ["select", "track", "compare", "inspect"],
    "interactive_entity": ["track", "select", "inspect"],
    "container_entity":   ["approach", "inspect"],
    "tracked_entity":     ["track", "select", "inspect"],
    "blocking_entity":    ["inspect", "track"],
    "navigable_region":   ["approach", "inspect"],
}

_AFFORD_DESKTOP: Dict[str, List[str]] = {
    "selectable_entity":  ["focus", "select", "inspect", "read"],
    "interactive_entity": ["focus", "open", "select"],
    "container_entity":   ["focus", "inspect"],
    "tracked_entity":     ["focus", "track"],
    "blocking_entity":    ["inspect"],
    "navigable_region":   ["focus", "select"],
}

_AFFORD_OMIT_ONTOLOGIES = {"textual_anchor", "goal_indicator"}

# ── Per-domain goal templates ─────────────────────────────────────────
# Used when the env doesn't carry a self-describing instruction.

_GAMINGAGENT_GOALS: Dict[str, str] = {
    "twenty_forty_eight": (
        "Reach the highest tile by merging matching tiles; avoid filling "
        "the board."
    ),
    "candy_crush": (
        "Clear level objectives by matching candies in groups of three or "
        "more."
    ),
    "tetris": (
        "Place tetromino pieces to clear lines and survive as long as "
        "possible."
    ),
    "tictactoe": "Place three of your marks in a row.",
    "texasholdem": "Maximise expected pot value across betting rounds.",
}

_GYMV_GENRE_GOAL: Dict[str, str] = {
    "shmup":      "Survive enemy waves and maximise score.",
    "platformer": "Reach the level exit while avoiding hazards.",
    "fighting":   "Defeat the opponent by depleting their health.",
    "puzzle":     "Match / align pieces to clear them.",
    "beatemup":   "Defeat enemies along a side-scrolling stage.",
    "action":     "Progress through stages while defeating enemies.",
    "racing":     "Finish the course in the fastest time.",
}


# ── Core entity row ───────────────────────────────────────────────────


@dataclass
class _Entity:
    eid: str
    type: str  # element | object | region | text
    label: str
    ontology: str = ""  # see vlm_wrapper.schema._SECTION_ENTITIES
    bid: Optional[str] = None
    pos: Optional[str] = None
    value: Optional[str] = None
    state: Optional[str] = None  # comma-joined sub-states, optional
    # Optional per-entity affordance override; if None the renderer
    # derives one from ``ontology`` via :data:`_AFFORD_GAME` /
    # :data:`_AFFORD_DESKTOP`.
    affords: Optional[List[str]] = None
    # Per-entity position uncertainty, mirrored into ``<uncertainty>``
    # rows when present.  None → entity skipped in uncertainty section.
    pos_uncertainty: Optional[str] = None  # "low" | "medium" | "high"


# ── Public entrypoint ─────────────────────────────────────────────────


def state_to_markup(
    *,
    obs_nl: str,
    info: Dict[str, Any],
    game: str,
    step: int,
    max_entities: int = DEFAULT_MAX_ENTITIES,
) -> str:
    """Render ``<state>...</state>`` from one env step's outputs.

    Parameters
    ----------
    obs_nl
        The natural-language observation string the actor would see.
    info
        The ``info`` dict returned alongside ``obs_nl`` from
        ``env.reset()`` / ``env.step()``.  Reads ``env_name``, ``game_name``,
        ``structured_state`` (when present), ``action_names``, ``score``,
        ``score_normalised``, ``raw_obs`` (for browsergym only).
    game
        The high-level game / task identifier the trainer uses (matches
        keys in ``GAME_CONFIGS`` etc.).
    step
        Zero-indexed environment step number.
    max_entities
        Hard cap on entities emitted.  Default 12 keeps prompts compact.

    Returns
    -------
    str
        A valid ``<state>...</state>`` block (always, even on errors).
    """
    try:
        env_name = (info.get("env_name") or "").lower()
        if env_name == "gymv_temporal":
            return _render_gymv(
                obs_nl=obs_nl, info=info, game=game,
                step=step, max_entities=max_entities,
            )
        if env_name == "gamingagent":
            return _render_gamingagent(
                obs_nl=obs_nl, info=info, game=game,
                step=step, max_entities=max_entities,
            )
        if env_name == "osworld":
            return _render_osworld(
                obs_nl=obs_nl, info=info, game=game,
                step=step, max_entities=max_entities,
            )
        if env_name == "orak":
            return _render_orak(
                obs_nl=obs_nl, info=info, game=game,
                step=step, max_entities=max_entities,
            )
        if env_name == "browsergym" or game.startswith("browsergym/"):
            return _render_browsergym(
                obs_nl=obs_nl, info=info, game=game,
                step=step, max_entities=max_entities,
            )
        return _render_fallback(
            obs_nl=obs_nl, info=info, game=game, step=step,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "state_to_markup failed (game=%s env=%s step=%d): %s; "
            "falling back to minimal markup",
            game, info.get("env_name"), step, exc,
        )
        return _render_fallback(
            obs_nl=obs_nl, info=info, game=game, step=step,
        )


# ── Common renderer ───────────────────────────────────────────────────


def _render_state_block(
    *,
    domain: str,
    task: str,
    goal: str,
    step: int,
    entities: List[_Entity],
    state_flags: Dict[str, str],
    target_eid: Optional[str],
    blocker_eid: Optional[str],
    candidate_set: List[str],
    actions: List[str],
    relations: Optional[List[str]] = None,
    constraint: Optional[str] = None,
    history_anchor: Optional[str] = None,
    afford_flavour: str = "game",
) -> str:
    """Emit a full-shape ``<state>...</state>`` block aligned with the
    cold-start VLM gold (see ``vlm_wrapper/few_shot_examples/*.txt``).

    Sections emitted (always, in this order):
        entities, attributes, affordances, relations,
        state_flags, targets, uncertainty, actions
    """
    afford_table = (
        _AFFORD_DESKTOP if afford_flavour == "desktop" else _AFFORD_GAME
    )

    lines: List[str] = ["<state>"]
    lines.append(f"domain={domain}")
    lines.append(f"task={_short(task, 80)}")
    lines.append(f"goal={_short(goal, 200)}")
    lines.append(f"step={step}")
    lines.append("")

    # ── <entities> ────────────────────────────────────────────────────
    lines.append("<entities>")
    for e in entities:
        bid_repr = e.bid if e.bid is not None else "null"
        pos_repr = e.pos if e.pos is not None else "null"
        ontology = e.ontology or "tracked_entity"
        lines.append(
            f"{e.eid}[type={e.type}, "
            f"label={_short(e.label, 60)}, "
            f"bid={bid_repr}, "
            f"pos={pos_repr}, "
            f"ontology={ontology}]"
        )
    lines.append("")

    # ── <attributes> ──────────────────────────────────────────────────
    lines.append("<attributes>")
    for e in entities:
        if e.value is not None:
            lines.append(f"{e.eid}.value={_short(str(e.value), 80)}")
        if e.state:
            lines.append(f"{e.eid}.state={e.state}")
    lines.append("")

    # ── <affordances> ─────────────────────────────────────────────────
    lines.append("<affordances>")
    for e in entities:
        if e.affords is not None:
            verbs = e.affords
        elif e.ontology in _AFFORD_OMIT_ONTOLOGIES:
            continue
        else:
            verbs = afford_table.get(e.ontology, [])
        if not verbs:
            continue
        lines.append(f"{e.eid}.affords=[{', '.join(verbs)}]")
    lines.append("")

    # ── <relations> ───────────────────────────────────────────────────
    lines.append("<relations>")
    for rel in (relations or []):
        lines.append(rel)
    lines.append("")

    # ── <state_flags> (always 6 fields, in the gold's canonical order) ─
    flags_full: Dict[str, str] = {
        "progress":      state_flags.get("progress", "0.0"),
        "phase":         state_flags.get("phase", _default_phase(step)),
        "scene_type":    state_flags.get(
            "scene_type", _default_scene_type(domain),
        ),
        "error":         state_flags.get("error", "null"),
        "dialog_open":   state_flags.get("dialog_open", "false"),
        "input_pending": state_flags.get("input_pending", "false"),
    }
    lines.append("<state_flags>")
    for k, v in flags_full.items():
        lines.append(f"{k}={_short(str(v), 80)}")
    # Also pass through any extra (non-canonical) flags so callers
    # can attach env-specific telemetry (e.g. ``last_reward``)
    for k, v in state_flags.items():
        if k not in flags_full:
            lines.append(f"{k}={_short(str(v), 80)}")
    lines.append("")

    # ── <targets> (always 5 fields) ───────────────────────────────────
    lines.append("<targets>")
    lines.append(f"target={target_eid or 'null'}")
    lines.append(f"blocker={blocker_eid or 'null'}")
    lines.append(f"constraint={_short(constraint or 'null', 120)}")
    lines.append(
        f"candidate_set=[{','.join(candidate_set)}]"
        if candidate_set else "candidate_set=[]"
    )
    lines.append(f"history_anchor={history_anchor or 'null'}")
    lines.append("")

    # ── <uncertainty> ─────────────────────────────────────────────────
    lines.append("<uncertainty>")
    for e in entities:
        if e.pos_uncertainty:
            lines.append(f"{e.eid}.pos={e.pos_uncertainty}")
    lines.append("")

    # ── <actions> ─────────────────────────────────────────────────────
    if actions:
        lines.append("<actions>")
        for i, a in enumerate(actions[:5], 1):
            lines.append(f"a{i}={_short(a, 50)}")

    lines.append("</state>")
    return "\n".join(lines)


def _default_phase(step: int) -> str:
    """Coarse phase bucketing matched to gold's ``early/mid/late`` usage."""
    if step < 10:
        return "early"
    if step < 50:
        return "mid"
    return "late"


def _default_scene_type(domain: str) -> str:
    """Domain-default scene_type mirroring values in the gold examples."""
    return {
        "gymv":    "game_play",
        "browser": "page_load",
        "desktop": "main_window",
    }.get(domain, "game_play")


def _format_task(domain_kind: str, raw_game: str) -> str:
    """Mirror cold-start ``task=`` formatting conventions.

    Cold-start gold uses prefixed identifiers (``Temporal/Strider-v0``,
    ``make_gaming_env/twenty_forty_eight``, ``make_orak_env/super_mario``,
    ``browsergym/miniwob.buy-ticket``).  If ``raw_game`` is already
    prefix-tagged we return it unchanged, otherwise we synthesise the
    correct prefix.  Pass ``domain_kind`` ∈ {``temporal``, ``gaming``,
    ``orak``, ``raw``}; ``raw`` returns the input untouched.
    """
    g = (raw_game or "").strip()
    if not g:
        return ""
    if "/" in g:
        return g
    if domain_kind == "temporal":
        return f"Temporal/{g}-v0"
    if domain_kind == "gaming":
        return f"make_gaming_env/{g}"
    if domain_kind == "orak":
        return f"make_orak_env/{g}"
    return g


def _short(s: Optional[str], n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else (s[: max(0, n - 1)].rstrip() + "…")


# ── Per-domain renderers ──────────────────────────────────────────────


def _render_gymv(
    *, obs_nl: str, info: Dict[str, Any], game: str,
    step: int, max_entities: int,
) -> str:
    ss: Dict[str, Any] = info.get("structured_state") or {}
    display_name = ss.get("display_name") or game
    genre = ss.get("genre") or "unknown"
    grounding_focus = ss.get("grounding_focus") or ""
    goal_template = _GYMV_GENRE_GOAL.get(
        genre, "Maximise score and progress.",
    )
    goal = (
        f"[{display_name}, {genre}] {goal_template}"
        + (f" Focus: {grounding_focus}." if grounding_focus else "")
    )

    entities: List[_Entity] = []
    src_entities = list(ss.get("entities") or [])[:max_entities]
    for src in src_entities:
        val = src.get("value")
        entities.append(_Entity(
            eid=src.get("eid", f"e{len(entities) + 1}"),
            type=src.get("type", "object"),
            label=src.get("label", ""),
            ontology=src.get("ontology", ""),
            bid=src.get("bid"),
            pos=src.get("pos"),
            value=str(val) if val is not None else None,
        ))

    sim = ss.get("simulation") or {}
    state_flags: Dict[str, str] = {
        "scene_type": "game_play",
    }
    if sim.get("episode_reward") is not None:
        state_flags["episode_reward"] = str(sim["episode_reward"])
    if sim.get("step_reward") is not None:
        state_flags["last_reward"] = str(sim["step_reward"])
    parsed = ss.get("parsed_text") or {}
    if parsed.get("gameover"):
        state_flags["error"] = f"gameover={parsed['gameover']}"

    available = (
        (ss.get("control") or {}).get("available_actions")
        or info.get("action_names")
        or []
    )
    actions = [str(a) for a in available][:5]

    goal_entities = [e for e in entities if e.ontology == "goal_indicator"]
    target_eid = (
        goal_entities[0].eid if goal_entities
        else (entities[0].eid if entities else None)
    )
    candidate_set = [e.eid for e in entities[:5]]

    return _render_state_block(
        domain="gymv",
        task=_format_task("temporal", display_name),
        goal=goal,
        step=step,
        entities=entities,
        state_flags=state_flags,
        target_eid=target_eid,
        blocker_eid=None,
        candidate_set=candidate_set,
        actions=actions,
        constraint=(grounding_focus or None),
        afford_flavour="game",
    )


def _render_gamingagent(
    *, obs_nl: str, info: Dict[str, Any], game: str,
    step: int, max_entities: int,
) -> str:
    ss: Dict[str, Any] = info.get("structured_state") or {}
    detected_game = info.get("game_name") or ss.get("game") or game
    goal = (
        _GAMINGAGENT_GOALS.get(game)
        or _GAMINGAGENT_GOALS.get(detected_game)
        or f"Achieve the win condition of {detected_game}."
    )

    entities: List[_Entity] = []
    eid = 1

    board = ss.get("board")
    if board:
        entities.append(_Entity(
            eid=f"e{eid}", type="region", label="board",
            ontology="container_entity",
            value=_short(str(board), 100),
        ))
        eid += 1

    tracked_fields = [
        ("max_tile",    "goal_indicator"),
        ("max_count",   "tracked_entity"),
        ("empty",       "tracked_entity"),
        ("merges",      "tracked_entity"),
        ("phase",       "tracked_entity"),
        ("self",        "tracked_entity"),
        ("objective",   "goal_indicator"),
        ("critical",    "blocking_entity"),
        ("progress",    "tracked_entity"),
    ]
    for key, ontology in tracked_fields:
        if key in ss and len(entities) < max_entities:
            entities.append(_Entity(
                eid=f"e{eid}", type="text", label=key,
                ontology=ontology, value=str(ss[key]),
            ))
            eid += 1

    state_flags: Dict[str, str] = {
        "scene_type": "game_play",
    }
    if "phase" in ss:
        state_flags["phase"] = str(ss["phase"])
    if "reward" in ss:
        state_flags["last_reward"] = str(ss["reward"])

    aff = ss.get("affordance") or ""
    actions = [a.strip() for a in aff.split(",") if a.strip()][:5]
    if not actions:
        actions = [str(a) for a in info.get("action_names") or []][:5]

    goal_entities = [e for e in entities if e.ontology == "goal_indicator"]
    target_eid = (
        goal_entities[0].eid if goal_entities
        else (entities[0].eid if entities else None)
    )
    blocker_entities = [
        e for e in entities if e.ontology == "blocking_entity"
    ]
    blocker_eid = blocker_entities[0].eid if blocker_entities else None
    candidate_set = [e.eid for e in entities[:5]]

    # Deterministic ``contains(board, *)`` relations when a board is the
    # first entity, mirroring the cold-start 2048/candy_crush gold.
    relations: List[str] = []
    if entities and entities[0].label == "board":
        for e in entities[1:]:
            if e.ontology in {
                "selectable_entity", "tracked_entity", "navigable_region",
            }:
                relations.append(f"contains({entities[0].eid},{e.eid})")

    return _render_state_block(
        domain="gymv",
        task=_format_task("gaming", str(detected_game)),
        goal=goal,
        step=step,
        entities=entities,
        state_flags=state_flags,
        target_eid=target_eid,
        blocker_eid=blocker_eid,
        candidate_set=candidate_set,
        actions=actions,
        relations=relations,
        afford_flavour="game",
    )


def _render_osworld(
    *, obs_nl: str, info: Dict[str, Any], game: str,
    step: int, max_entities: int,
) -> str:
    ss: Dict[str, Any] = info.get("structured_state") or {}
    instruction = ss.get("instruction") or ""
    goal = instruction or f"Complete the desktop task: {game}."

    entities: List[_Entity] = [_Entity(
        eid="e1", type="region", label="screen",
        ontology="navigable_region",
    )]
    eid = 2

    for region_name, region_desc in (ss.get("screen_regions") or {}).items():
        if len(entities) >= max_entities:
            break
        entities.append(_Entity(
            eid=f"e{eid}", type="region", label=str(region_name),
            ontology="container_entity",
            value=_short(str(region_desc), 60),
        ))
        eid += 1

    if ss.get("ui_element_count") is not None and len(entities) < max_entities:
        entities.append(_Entity(
            eid=f"e{eid}", type="text", label="ui_element_count",
            ontology="tracked_entity",
            value=str(ss["ui_element_count"]),
        ))
        eid += 1

    if ss.get("terminal_lines") and len(entities) < max_entities:
        entities.append(_Entity(
            eid=f"e{eid}", type="text", label="terminal_last",
            ontology="textual_anchor",
            value=_short(str(ss.get("terminal_last", "")), 60),
        ))
        eid += 1

    state_flags: Dict[str, str] = {
        "scene_type": "main_window",
        "dialog_open": "true" if ss.get("has_dialog") else "false",
    }
    if ss.get("last_action"):
        state_flags["last_action"] = _short(str(ss["last_action"]), 60)

    actions = [
        "pyautogui.click(x,y)",
        "pyautogui.typewrite('text')",
        "pyautogui.hotkey('ctrl','c')",
        "DONE",
        "WAIT",
    ]

    candidate_set = [e.eid for e in entities[:5]]
    # Deterministic ``contains(screen, *)`` relations matching desktop gold.
    relations: List[str] = []
    if entities and entities[0].label == "screen":
        for e in entities[1:]:
            if e.type == "region":
                relations.append(f"contains({entities[0].eid},{e.eid})")

    return _render_state_block(
        domain="desktop",
        task=str(game),
        goal=goal,
        step=step,
        entities=entities,
        state_flags=state_flags,
        target_eid="e1",
        blocker_eid=None,
        candidate_set=candidate_set,
        actions=actions,
        relations=relations,
        constraint=(_short(instruction, 120) if instruction else None),
        afford_flavour="desktop",
    )


def _render_orak(
    *, obs_nl: str, info: Dict[str, Any], game: str,
    step: int, max_entities: int,
) -> str:
    # Orak ships no structured_state today; this renderer is degraded by
    # design.  See the next-milestone TODO ("补 orak structured_state 头").
    game_name = info.get("game_name") or game
    task_text = info.get("task") or game_name
    goal = f"[orak/{game_name}] {_short(str(task_text), 160)}"

    entities: List[_Entity] = [
        _Entity(
            eid="e1", type="region", label="game_screen",
            ontology="navigable_region",
        ),
        _Entity(
            eid="e2", type="text", label="raw_observation",
            ontology="textual_anchor",
            value=_short(obs_nl, 200),
        ),
    ]
    if info.get("score") is not None:
        entities.append(_Entity(
            eid="e3", type="text", label="score",
            ontology="goal_indicator", value=str(info["score"]),
        ))

    state_flags: Dict[str, str] = {
        "scene_type": "game_play",
    }
    if info.get("score_normalised") is not None:
        try:
            progress = float(info["score_normalised"]) / 100.0
            state_flags["progress"] = f"{max(0.0, min(1.0, progress)):.2f}"
        except (TypeError, ValueError):
            pass

    actions = [str(a) for a in info.get("action_names") or []][:5]
    target_eid = "e3" if len(entities) >= 3 else "e1"
    return _render_state_block(
        domain="gymv",
        task=_format_task("orak", str(game_name)),
        goal=goal,
        step=step,
        entities=entities,
        state_flags=state_flags,
        target_eid=target_eid,
        blocker_eid=None,
        candidate_set=[e.eid for e in entities],
        actions=actions,
        afford_flavour="game",
    )


def _render_browsergym(
    *, obs_nl: str, info: Dict[str, Any], game: str,
    step: int, max_entities: int,
) -> str:
    raw_obs = info.get("raw_obs")
    if isinstance(raw_obs, dict) and raw_obs.get("axtree_object"):
        try:
            from browsergym_wrapper.heuristic import obs_to_schema
            return obs_to_schema(
                raw_obs, step=step, task_id=game, max_entities=max_entities,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "browsergym obs_to_schema failed for game=%s step=%d: %s",
                game, step, exc,
            )
    return _render_fallback(
        obs_nl=obs_nl, info=info, game=game, step=step,
    )


def _render_fallback(
    *, obs_nl: str, info: Dict[str, Any], game: str, step: int,
) -> str:
    entities: List[_Entity] = [
        _Entity(
            eid="e1", type="region", label="environment",
            ontology="navigable_region",
        ),
        _Entity(
            eid="e2", type="text", label="observation",
            ontology="textual_anchor",
            value=_short(obs_nl, 240) if obs_nl else "(empty)",
        ),
    ]
    actions = [str(a) for a in info.get("action_names") or []][:5]
    return _render_state_block(
        domain="unknown",
        task=str(game),
        goal=f"Interact with {game}.",
        step=step,
        entities=entities,
        state_flags={},  # all 6 canonical fields are auto-defaulted
        target_eid="e1",
        blocker_eid=None,
        candidate_set=["e1", "e2"],
        actions=actions,
        afford_flavour="game",
    )


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_MAX_ENTITIES",
    "state_to_markup",
]
