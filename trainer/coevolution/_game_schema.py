"""Phase-start ``GameProfile`` generator — 1 × 35B call per (phase, game).

Goal
----
Inject a *static, semantic* description of the current game / task into the
actor's system prompt **once per phase**, so that:

* the actor knows what "winning" looks like (goal, win-signal, hazards,
  failure modes) instead of seeing only the raw obs.text + a numeric ACTION
  prompt,
* the LLM Crafter (Path 2) and LLM Harness validator (Path 4) get a free
  per-game few-shot exemplar (the ``state_example_markup`` field) without
  burning their own 35B budget on style alignment,
* the trainer pays the 35B cost **once per phase boundary**, not per step
  — so the marginal cost over a 100-step phase × 8-episode rollout is one
  ~1k-token call, ~10-30 s of latency at phase start.

Pipeline contract
-----------------
1. The orchestrator detects a curriculum transition (``config.games``
   changed) and calls :func:`ensure_game_schemas`.
2. For each game in the new phase that does not have a cached profile,
   we:

   a. spin up a 1-step env via the same ``make_env`` path the trainer
      uses, take the first frame's ``obs_nl`` + ``info``,
   b. render a deterministic ``<state>`` markup via
      :mod:`trainer.coevolution._state_to_markup`,
   c. fire one ``ask_model`` call to the 35B
      ``BACKBONE_JUDGE_MODEL`` endpoint (same routing as the LLM
      promotion judge — ``VLLM_BASE_URL_MAP`` → port 8004),
   d. parse the response into a :class:`GameProfile` and persist it
      under ``run_dir/phase_artifacts/<game>.schema.json``.

3. On any failure (LLM error, parse error, env failure) we degrade to
   :func:`_minimum_fallback_profile` so the phase never blocks on a
   flaky 35B endpoint.

Output contract
---------------
Two artefacts in one 35B response:

* ``=== GAME_PROFILE ===`` — compact (~150-token) goal / win_signal /
  hazards / recurring_entities / key_actions / failure_modes — gets
  rendered by :func:`render_for_actor_prompt` and prepended to the
  actor's SYSTEM_PROMPT and SKILL_SELECTION_SYSTEM_PROMPT.
* ``=== STATE_EXAMPLE ===`` — full ``<state>...</state>`` markup
  exemplar — cached so Path 2 / Path 4 can reuse as few-shot anchor
  without firing their own 35B calls.

Cache invalidation
------------------
The cache key includes ``SCHEMA_VERSION`` (markup vocab) plus a SHA-1
digest of ``available_actions`` and any ``ram_watch`` / ``parsed_text``
keys observable on the first frame.  A different game version (different
RAM layout / action set) thus produces a different key and forces a
fresh 35B call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from trainer.coevolution._state_to_markup import (
    SCHEMA_VERSION as MARKUP_SCHEMA_VERSION,
    state_to_markup,
)

logger = logging.getLogger(__name__)

# Latch to MARKUP_SCHEMA_VERSION so cache stale-ness is detected when the
# vocabulary changes.  Bump independently when the GameProfile *prompt*
# format itself changes (i.e. force-regenerate even with same markup).
PROFILE_VERSION = "p1.0"

# Hard caps to keep prompt + cache footprint bounded.
_MAX_HAZARDS = 5
_MAX_RECURRING_ENTITIES = 8
_MAX_KEY_ACTIONS = 8
_MAX_FAILURE_MODES = 5
_MAX_FREE_TEXT_CHARS = 200
_MAX_STATE_EXAMPLE_CHARS = 2000


# ── Output dataclass ──────────────────────────────────────────────────


@dataclass
class GameProfile:
    """Static semantic profile of one game / task.

    All free-text fields are short (≤200 chars).  Lists are capped at the
    constants above so the rendered actor prompt fits in a few hundred
    tokens regardless of which game we're on.
    """

    game: str
    display_name: str
    genre: str
    goal: str
    win_signal: str
    hazards: List[str] = field(default_factory=list)
    recurring_entities: List[str] = field(default_factory=list)
    key_actions: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    state_example_markup: str = ""  # full <state>...</state> exemplar

    cache_key: str = ""
    schema_version: str = PROFILE_VERSION
    markup_version: str = MARKUP_SCHEMA_VERSION
    generated_at: float = 0.0
    source: str = "llm"  # "llm" | "fallback" | "cache"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GameProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


# ── Cache I/O ─────────────────────────────────────────────────────────


def _artifact_dir(run_dir: str) -> Path:
    p = Path(run_dir) / "phase_artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_path(run_dir: str, game: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", game)
    return _artifact_dir(run_dir) / f"{safe}.schema.json"


def _make_cache_key(*, game: str, info: Dict[str, Any]) -> str:
    """SHA-1 over (markup-version, profile-version, action set, RAM keys).

    Stable across runs for the same env build; changes if the env layout
    changes (different RAM watch / action mapping / structured-state
    schema) — forcing a fresh 35B call.
    """
    ss = info.get("structured_state") or {}
    ram_keys = sorted((ss.get("ram_watch") or {}).keys())
    parsed_keys = sorted((ss.get("parsed_text") or {}).keys())
    available = (
        (ss.get("control") or {}).get("available_actions")
        or info.get("action_names")
        or []
    )
    fingerprint = "|".join([
        f"game={game}",
        f"markup={MARKUP_SCHEMA_VERSION}",
        f"profile={PROFILE_VERSION}",
        f"actions={','.join(sorted(str(a) for a in available))}",
        f"ram={','.join(ram_keys)}",
        f"parsed={','.join(parsed_keys)}",
        f"env={info.get('env_name', '')}",
    ])
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]


def load_cached(*, run_dir: str, game: str, expected_key: str) -> Optional[GameProfile]:
    path = _cache_path(run_dir, game)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("game_schema cache read failed (%s): %s", path, exc)
        return None
    if data.get("cache_key") != expected_key:
        logger.info(
            "game_schema cache stale for game=%s (have=%s want=%s); "
            "regenerating",
            game, data.get("cache_key"), expected_key,
        )
        return None
    if data.get("markup_version") != MARKUP_SCHEMA_VERSION:
        logger.info(
            "game_schema markup version drift for game=%s "
            "(have=%s want=%s); regenerating",
            game, data.get("markup_version"), MARKUP_SCHEMA_VERSION,
        )
        return None
    profile = GameProfile.from_dict(data)
    profile.source = "cache"
    return profile


def save_cached(*, run_dir: str, game: str, profile: GameProfile) -> None:
    path = _cache_path(run_dir, game)
    try:
        path.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("game_schema cache write failed (%s): %s", path, exc)


# ── Prompt construction & response parsing ────────────────────────────


_PROMPT_HEADER = """\
You are a game-analysis assistant.  You will be given the **first frame**
of one episode of a game (already rendered into a structured `<state>`
schema), plus static metadata about the game.  Output a short, static
profile of the game that an action-selecting agent can read once at the
start of each phase.  Be precise; the agent will rely on the win_signal
and hazards lines verbatim to choose actions.

Your output MUST contain exactly two blocks, separated by the markers
shown below.  Do not output anything before, between (besides the marker
line), or after these blocks.

=== GAME_PROFILE ===
goal: <one sentence describing what the player must do to make progress>
win_signal: <which observable quantity increases when the player is doing well — must reference an entity label or attribute that already appears in the <state> schema below; one short phrase>
hazards: [<≤{max_hazards} short noun phrases of things that hurt / penalise / kill the player>]
recurring_entities: [<≤{max_recurring} short noun phrases for entities the agent is likely to see repeatedly across steps; must be plausible given the entities in the <state> schema below>]
key_actions: [<≤{max_actions} actions from the available_actions list, ordered by typical usefulness>]
failure_modes: [<≤{max_failures} short phrases describing common ways the agent could lose progress or get stuck>]

=== STATE_EXAMPLE ===
<state>
... a polished version of the schema below, lightly enriched with
plausible <attributes>, <state_flags>, and <targets> values for this
frame.  Keep entity ids and labels exactly as given.  Do not invent
entities that are not in the input schema.  Limit to the same
domain= and task= values.  Output ≤{max_state_chars} characters.
</state>

Constraints:
- Use only entity ids / labels / actions that appear in the input schema.
- Lists must be JSON-style: [..,..]; if empty, write [].
- No commentary, no markdown headers, no emoji.
"""


def _build_prompt(
    *,
    game: str,
    info: Dict[str, Any],
    obs_nl: str,
    step: int,
) -> str:
    """Construct the 35B prompt.  Always returns a string."""
    markup = state_to_markup(
        obs_nl=obs_nl, info=info, game=game, step=step,
    )

    ss = info.get("structured_state") or {}
    metadata_lines = [
        f"game_id: {game}",
        f"env_name: {info.get('env_name', 'unknown')}",
        f"display_name: {ss.get('display_name', game)}",
        f"genre: {ss.get('genre', 'unknown')}",
    ]
    if ss.get("grounding_focus"):
        metadata_lines.append(f"grounding_focus: {ss['grounding_focus']}")
    if info.get("task"):
        metadata_lines.append(
            f"task_hint: {str(info['task'])[:_MAX_FREE_TEXT_CHARS]}"
        )
    available = (
        (ss.get("control") or {}).get("available_actions")
        or info.get("action_names")
        or []
    )
    if available:
        metadata_lines.append(
            "available_actions: ["
            + ", ".join(str(a) for a in available[:20]) + "]"
        )

    header = _PROMPT_HEADER.format(
        max_hazards=_MAX_HAZARDS,
        max_recurring=_MAX_RECURRING_ENTITIES,
        max_actions=_MAX_KEY_ACTIONS,
        max_failures=_MAX_FAILURE_MODES,
        max_state_chars=_MAX_STATE_EXAMPLE_CHARS,
    )

    return (
        f"{header}\n\n"
        f"--- GAME METADATA ---\n"
        + "\n".join(metadata_lines)
        + "\n\n--- FIRST FRAME SCHEMA ---\n"
        + markup
        + "\n\n--- FIRST FRAME RAW OBSERVATION TEXT (truncated) ---\n"
        + (obs_nl or "")[:600]
    )


# Robust to noisy LLM output (fenced blocks, stray whitespace).
_GAME_PROFILE_RE = re.compile(
    r"=== GAME_PROFILE ===\s*(.*?)(?:===\s*STATE_EXAMPLE\s*===|\Z)",
    re.DOTALL,
)
_STATE_EXAMPLE_RE = re.compile(
    r"=== STATE_EXAMPLE ===\s*(.*)", re.DOTALL,
)
_LIST_FIELD_RE = re.compile(r"^\s*([a-z_]+)\s*:\s*\[(.*)\]\s*$", re.IGNORECASE)
_SCALAR_FIELD_RE = re.compile(r"^\s*([a-z_]+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_STATE_BLOCK_RE = re.compile(r"<state>.*?</state>", re.DOTALL)


def _split_list_items(raw: str) -> List[str]:
    """Parse a JSON-ish list body into a clean list of short strings."""
    raw = raw.strip()
    if not raw:
        return []
    # Try JSON first.
    try:
        parsed = json.loads("[" + raw + "]")
        if isinstance(parsed, list):
            return [
                str(x).strip().strip("\"'")[:_MAX_FREE_TEXT_CHARS]
                for x in parsed if str(x).strip()
            ]
    except Exception:  # noqa: BLE001
        pass
    # Fallback: comma-separated, strip quotes/brackets.
    items: List[str] = []
    for chunk in raw.split(","):
        c = chunk.strip().strip("\"'`").strip()
        if c:
            items.append(c[:_MAX_FREE_TEXT_CHARS])
    return items


def _parse_response(
    raw: str,
    *,
    game: str,
    display_name: str,
    genre: str,
) -> Tuple[Optional[GameProfile], Optional[str]]:
    """Parse 35B output into a partial :class:`GameProfile`.

    Returns ``(profile, error_str)``.  ``profile`` is ``None`` when the
    response is unrecoverable; ``error_str`` is non-empty when *any*
    required field went missing (so the caller can log).
    """
    if not raw:
        return None, "empty_response"

    m_profile = _GAME_PROFILE_RE.search(raw)
    if not m_profile:
        return None, "missing_GAME_PROFILE_marker"

    profile_body = m_profile.group(1).strip()
    fields: Dict[str, Any] = {}
    for line in profile_body.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m_list = _LIST_FIELD_RE.match(line)
        if m_list:
            key = m_list.group(1).lower()
            fields[key] = _split_list_items(m_list.group(2))
            continue
        m_scalar = _SCALAR_FIELD_RE.match(line)
        if m_scalar:
            key = m_scalar.group(1).lower()
            val = m_scalar.group(2).strip().strip("\"'")
            fields[key] = val[:_MAX_FREE_TEXT_CHARS]

    goal = fields.get("goal") or ""
    win_signal = fields.get("win_signal") or ""
    if not goal or not win_signal:
        return None, "missing_goal_or_win_signal"

    hazards = fields.get("hazards") or []
    recurring = fields.get("recurring_entities") or []
    key_actions = fields.get("key_actions") or []
    failure_modes = fields.get("failure_modes") or []
    if not isinstance(hazards, list):
        hazards = []
    if not isinstance(recurring, list):
        recurring = []
    if not isinstance(key_actions, list):
        key_actions = []
    if not isinstance(failure_modes, list):
        failure_modes = []

    state_example = ""
    m_state = _STATE_EXAMPLE_RE.search(raw)
    if m_state:
        block_match = _STATE_BLOCK_RE.search(m_state.group(1))
        if block_match:
            state_example = block_match.group(0)[:_MAX_STATE_EXAMPLE_CHARS]

    return GameProfile(
        game=game,
        display_name=display_name,
        genre=genre,
        goal=goal,
        win_signal=win_signal,
        hazards=hazards[:_MAX_HAZARDS],
        recurring_entities=recurring[:_MAX_RECURRING_ENTITIES],
        key_actions=key_actions[:_MAX_KEY_ACTIONS],
        failure_modes=failure_modes[:_MAX_FAILURE_MODES],
        state_example_markup=state_example,
    ), None


# ── LLM call (sync, runs in executor) ─────────────────────────────────


def _ask_judge(
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import time as _t
    from API_func import ask_model
    _t0 = _t.monotonic()
    try:
        return ask_model(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ) or ""
    finally:
        try:
            from trainer.coevolution._run_loggers import (  # noqa: WPS433
                record_component_call,
            )
            record_component_call(
                "schema.profile",
                latency_ms=(_t.monotonic() - _t0) * 1000.0,
            )
        except Exception:  # noqa: BLE001
            pass


# ── Public API: per-game generation and ensure-loop ───────────────────


def _minimum_fallback_profile(
    *,
    game: str,
    info: Dict[str, Any],
    cache_key: str,
    reason: str,
) -> GameProfile:
    """Tiny deterministic profile so the phase never blocks on LLM flakiness."""
    ss = info.get("structured_state") or {}
    display_name = ss.get("display_name") or game
    genre = ss.get("genre") or "unknown"
    grounding = ss.get("grounding_focus") or ""

    available = (
        (ss.get("control") or {}).get("available_actions")
        or info.get("action_names")
        or []
    )
    key_actions = [str(a) for a in available[:_MAX_KEY_ACTIONS]]
    goal = (
        (info.get("task") or "")
        or f"Play {display_name} ({genre})"
        + (f" — {grounding}" if grounding else "")
    )[:_MAX_FREE_TEXT_CHARS]

    win_signal = "score increases or progress flag advances"
    if ss.get("entities"):
        for e in ss["entities"]:
            if e.get("ontology") == "goal_indicator" and e.get("label"):
                win_signal = (
                    f"{e['label']} value increases (goal_indicator)"
                )
                break

    profile = GameProfile(
        game=game,
        display_name=str(display_name),
        genre=str(genre),
        goal=goal,
        win_signal=win_signal,
        hazards=[],
        recurring_entities=[],
        key_actions=key_actions,
        failure_modes=[],
        state_example_markup="",
        cache_key=cache_key,
        generated_at=time.time(),
        source=f"fallback:{reason}",
    )
    return profile


async def generate_for_game(
    *,
    game: str,
    info: Dict[str, Any],
    obs_nl: str,
    model: str,
    executor: Optional[Any] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
) -> GameProfile:
    """Run one 35B call and return a :class:`GameProfile`.

    Always returns a profile.  On any failure path the result has
    ``source="fallback:<reason>"`` and the LLM-derived fields default
    to deterministic content.
    """
    ss = info.get("structured_state") or {}
    display_name = ss.get("display_name") or game
    genre = ss.get("genre") or "unknown"
    cache_key = _make_cache_key(game=game, info=info)

    prompt = _build_prompt(
        game=game, info=info, obs_nl=obs_nl, step=0,
    )

    raw = ""
    err: Optional[str] = None
    try:
        loop = asyncio.get_running_loop()
        raw = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                lambda: _ask_judge(
                    prompt=prompt, model=model,
                    temperature=temperature, max_tokens=max_tokens,
                ),
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        err = f"timeout_after_{timeout_s}s"
        logger.warning("game_schema 35B timeout for game=%s after %.1fs", game, timeout_s)
    except Exception as exc:  # noqa: BLE001
        err = f"call_failed:{type(exc).__name__}:{str(exc)[:80]}"
        logger.warning("game_schema 35B call failed for game=%s: %s", game, exc)

    if err is not None:
        profile = _minimum_fallback_profile(
            game=game, info=info, cache_key=cache_key, reason=err,
        )
        return profile

    profile, parse_err = _parse_response(
        raw, game=game,
        display_name=str(display_name), genre=str(genre),
    )
    if profile is None:
        logger.warning(
            "game_schema parse failure for game=%s err=%s raw=%r",
            game, parse_err, (raw or "")[:200],
        )
        profile = _minimum_fallback_profile(
            game=game, info=info, cache_key=cache_key,
            reason=f"parse_failed:{parse_err}",
        )
        return profile

    profile.cache_key = cache_key
    profile.markup_version = MARKUP_SCHEMA_VERSION
    profile.generated_at = time.time()
    profile.source = "llm"
    return profile


def default_env_factory(game: str, max_steps: int = 8) -> Any:
    """Build a one-shot env mirroring ``episode_runner.run_episode_async``.

    Used by :func:`ensure_game_schemas` to fetch a single first-frame
    ``(obs_nl, info)`` pair for prompt construction.  Callers are
    expected to call ``env.reset()`` and then ``env.close()``.

    Dispatch matches the trainer's runtime env construction:

    * ``GYMV_TEMPORAL_GAMES_SET`` → ``SubprocessEnv(env_kind="gymv")``
      (per-process emulator-singleton isolation; same wrapper used at
      training time so ``info["structured_state"]`` matches),
    * ``ORAK_GAMES_SET``         → ``SubprocessEnv`` or ``make_orak_env``,
    * everything else            → ``make_gaming_env`` +
      ``GamingAgentNLWrapper`` (with ``TetrisMacroActionWrapper`` for
      ``tetris``).

    Imports lazily so this module stays cheap to import outside the
    trainer.  Any unrecognised game raises ``ValueError`` so a typo in
    a curriculum file fails fast at phase boundary instead of producing
    a confusing fallback profile.
    """
    from trainer.coevolution.episode_runner import (
        _lazy_imports,
        GYMV_TEMPORAL_GAMES_SET,
        ORAK_GAMES_SET,
    )
    imp = _lazy_imports()
    SubprocessEnv = imp["SubprocessEnv"]
    GamingAgentNLWrapper = imp["GamingAgentNLWrapper"]
    make_gaming_env = imp["make_gaming_env"]
    make_orak_env = imp.get("make_orak_env")

    if game in GYMV_TEMPORAL_GAMES_SET:
        return SubprocessEnv(
            game=game, max_steps=max_steps, env_kind="gymv",
        )
    if game in ORAK_GAMES_SET:
        if make_orak_env is None:
            return SubprocessEnv(game=game, max_steps=max_steps)
        return make_orak_env(game, max_steps=max_steps)

    base = make_gaming_env(game=game, max_steps=max_steps)
    if game == "tetris":
        from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
        return TetrisMacroActionWrapper(GamingAgentNLWrapper(base))
    return GamingAgentNLWrapper(base)


async def ensure_game_schemas(
    *,
    games: List[str],
    run_dir: str,
    model: str,
    env_factory: Optional[Any] = None,  # callable: (game) -> env
    executor: Optional[Any] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout_s: float = 60.0,
    overwrite: bool = False,
) -> Dict[str, GameProfile]:
    """Ensure every game in *games* has a fresh ``GameProfile``.

    Sequential by design — phase boundary is the only call site, so we
    don't need parallelism, and 35B endpoint is shared with the
    promotion judge.
    """
    factory = env_factory if env_factory is not None else default_env_factory
    out: Dict[str, GameProfile] = {}
    for game in games:
        env = None
        try:
            env = factory(game)
            obs_nl, info = env.reset()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "game_schema env reset failed for game=%s: %s — "
                "writing minimal fallback",
                game, exc,
            )
            cache_key = _make_cache_key(
                game=game, info={"structured_state": {}, "action_names": []},
            )
            out[game] = _minimum_fallback_profile(
                game=game, info={}, cache_key=cache_key,
                reason=f"env_reset_failed:{type(exc).__name__}",
            )
            continue
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass

        cache_key = _make_cache_key(game=game, info=info)
        if not overwrite:
            cached = load_cached(
                run_dir=run_dir, game=game, expected_key=cache_key,
            )
            if cached is not None:
                logger.info(
                    "game_schema cache HIT  game=%s key=%s source=%s",
                    game, cache_key, cached.source,
                )
                out[game] = cached
                continue

        t0 = time.monotonic()
        profile = await generate_for_game(
            game=game, info=info, obs_nl=obs_nl,
            model=model, executor=executor,
            max_tokens=max_tokens, temperature=temperature,
            timeout_s=timeout_s,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            "game_schema generated game=%s source=%s elapsed=%.1fs "
            "key=%s goal=%r",
            game, profile.source, elapsed, profile.cache_key,
            profile.goal[:80],
        )
        save_cached(run_dir=run_dir, game=game, profile=profile)
        out[game] = profile
    return out


# ── Renderers (consumed by orchestrator / episode_runner) ─────────────


def render_for_actor_prompt(profile: GameProfile) -> str:
    """Compact ~150-token block for actor SYSTEM_PROMPT injection."""
    parts: List[str] = [
        f"GAME PROFILE — {profile.display_name} ({profile.genre})",
        f"Goal: {profile.goal}",
        f"Win signal: {profile.win_signal}",
    ]
    if profile.hazards:
        parts.append(
            "Hazards: " + "; ".join(profile.hazards)
        )
    if profile.recurring_entities:
        parts.append(
            "Recurring entities: " + ", ".join(profile.recurring_entities)
        )
    if profile.key_actions:
        parts.append(
            "Key actions: " + ", ".join(profile.key_actions)
        )
    if profile.failure_modes:
        parts.append(
            "Failure modes to avoid: " + "; ".join(profile.failure_modes)
        )
    return "\n".join(parts)


def render_state_example(profile: GameProfile) -> str:
    """Full ``<state>`` exemplar for Path 2 / Path 4 few-shot use."""
    return profile.state_example_markup or ""


__all__ = [
    "PROFILE_VERSION",
    "GameProfile",
    "default_env_factory",
    "generate_for_game",
    "ensure_game_schemas",
    "load_cached",
    "save_cached",
    "render_for_actor_prompt",
    "render_state_example",
]
