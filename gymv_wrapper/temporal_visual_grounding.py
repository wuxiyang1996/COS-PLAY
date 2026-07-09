"""Per-game visual grounding for Gym-V Temporal (stable-retro / Genesis) envs.

Builds a JSON-serializable **visual grounding schema** from each multimodal
``Observation``: frame geometry, parsed ``obs.text``, ``metadata`` from
:class:`gym_v.envs.multi_turn.temporal.retro_env.RetroGymVEnv` (RAM watch,
actions, rewards), plus light game metadata.

Use :class:`TemporalVisualGroundingWrapper` (or a generated per-title subclass)
to attach ``obs.metadata["visual_grounding"]`` on every step.  Pixel-level
object detection is **not** performed here — the schema is a structured fuse
of **visual dimensions** + **ground-truth simulator signals** so VLMs and
tool agents share one JSON contract per game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from gymv_wrapper.heuristic import parse_retro_pipe_text


@dataclass(frozen=True)
class GameVisualSpec:
    """Static hints for a registered ``Temporal/<Title>-v0`` environment."""

    gym_env_id: str
    retro_game: str
    display_name: str
    genre: str
    grounding_focus: str


# Mirrors gym_v.envs.__init__ Temporal registrations (Genesis via stable-retro).
TEMPORAL_GAME_SPECS: dict[str, GameVisualSpec] = {
    "Temporal/Airstriker-v0": GameVisualSpec(
        "Temporal/Airstriker-v0",
        "Airstriker-Genesis-v0",
        "Airstriker",
        "shmup",
        "ships, bullets, scrolling playfield",
    ),
    "Temporal/AlteredBeast-v0": GameVisualSpec(
        "Temporal/AlteredBeast-v0",
        "AlteredBeast-Genesis-v0",
        "Altered Beast",
        "beatemup",
        "player avatar, beasts, power orbs",
    ),
    "Temporal/CastleOfIllusion-v0": GameVisualSpec(
        "Temporal/CastleOfIllusion-v0",
        "CastleOfIllusion-Genesis-v0",
        "Castle of Illusion",
        "platformer",
        "mickey, platforms, collectibles",
    ),
    "Temporal/CastlevaniaBloodlines-v0": GameVisualSpec(
        "Temporal/CastlevaniaBloodlines-v0",
        "CastlevaniaBloodlines-Genesis-v0",
        "Castlevania Bloodlines",
        "action",
        "whip combat, enemies, verticality",
    ),
    "Temporal/Columns-v0": GameVisualSpec(
        "Temporal/Columns-v0",
        "Columns-Genesis-v0",
        "Columns",
        "puzzle",
        "jewel stacks, falling column, match clears",
    ),
    "Temporal/DynamiteHeaddy-v0": GameVisualSpec(
        "Temporal/DynamiteHeaddy-v0",
        "DynamiteHeaddy-Genesis-v0",
        "Dynamite Headdy",
        "platformer",
        "detachable head, bosses, platforms",
    ),
    "Temporal/GoldenAxe-v0": GameVisualSpec(
        "Temporal/GoldenAxe-v0",
        "GoldenAxe-Genesis-v0",
        "Golden Axe",
        "beatemup",
        "melee spacing, mounts, magic",
    ),
    "Temporal/KidChameleon-v0": GameVisualSpec(
        "Temporal/KidChameleon-v0",
        "KidChameleon-Genesis-v0",
        "Kid Chameleon",
        "platformer",
        "helmets, transformations, hazards",
    ),
    "Temporal/MortalKombatII-v0": GameVisualSpec(
        "Temporal/MortalKombatII-v0",
        "MortalKombatII-Genesis-v0",
        "Mortal Kombat II",
        "fighting",
        "two fighters, health, special moves",
    ),
    "Temporal/SpaceHarrierII-v0": GameVisualSpec(
        "Temporal/SpaceHarrierII-v0",
        "SpaceHarrierII-Genesis-v0",
        "Space Harrier II",
        "shmup",
        "pseudo-3d rail, obstacles, projectiles",
    ),
    "Temporal/StreetsOfRage2-v0": GameVisualSpec(
        "Temporal/StreetsOfRage2-v0",
        "StreetsOfRage2-Genesis-v0",
        "Streets of Rage 2",
        "beatemup",
        "combo chains, throws, crowd control",
    ),
    "Temporal/Strider-v0": GameVisualSpec(
        "Temporal/Strider-v0",
        "Strider-Genesis-v0",
        "Strider",
        "action",
        "slash mobility, large sprites, bosses",
    ),
    "Temporal/ThunderForceIII-v0": GameVisualSpec(
        "Temporal/ThunderForceIII-v0",
        "ThunderForceIII-Genesis-v0",
        "Thunder Force III",
        "shmup",
        "weapon pickups, speed, bullet patterns",
    ),
}


def _image_size(obs: Any) -> dict[str, int | None]:
    img = getattr(obs, "image", None)
    if img is None:
        return {"width": None, "height": None, "channels": None}
    if isinstance(img, list) and img:
        img = img[-1]
    try:
        w, h = img.size
    except Exception:
        return {"width": None, "height": None, "channels": None}
    ch = 4 if getattr(img, "mode", "") == "RGBA" else 3
    return {"width": int(w), "height": int(h), "channels": ch}


def _entities_from_retro(
    parsed: Mapping[str, str],
    ram_watch: Mapping[str, Any],
    spec: GameVisualSpec | None,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    eid = 1
    entities.append({
        "eid": f"e{eid}",
        "type": "region",
        "label": "genesis_viewport",
        "ontology": "navigable_region",
        "bid": None,
        "pos": None,
        "source": "visual_frame",
        "notes": spec.grounding_focus if spec else "retro playfield",
    })
    eid += 1
    for key in ("score", "lives", "health", "level", "gameover"):
        if key in parsed:
            entities.append({
                "eid": f"e{eid}",
                "type": "text",
                "label": f"hud_{key}",
                "ontology": "goal_indicator" if key in ("score", "level") else "tracked_entity",
                "bid": None,
                "pos": None,
                "source": "obs_text",
                "value": parsed[key],
            })
            eid += 1
    for k, v in ram_watch.items():
        if k in ("score", "lives", "health", "level", "gameover"):
            continue
        try:
            val_repr: str | int | float = v.item() if hasattr(v, "item") else v  # numpy scalars
        except Exception:
            val_repr = str(v)
        entities.append({
            "eid": f"e{eid}",
            "type": "object",
            "label": f"ram_{k}",
            "ontology": "tracked_entity",
            "bid": None,
            "pos": None,
            "source": "ram_watch",
            "value": val_repr,
        })
        eid += 1
    return entities


def build_temporal_visual_schema(env_id: str, obs: Any) -> dict[str, Any]:
    """Fuse frame layout + RetroGymVEnv text/metadata into one JSON schema.

    Parameters
    ----------
    env_id
        Registered Gym-V id, e.g. ``"Temporal/Airstriker-v0"``.
    obs
        Any object with ``.text``, ``.metadata`` (and optionally ``.image``).
    """
    meta: dict[str, Any] = dict(getattr(obs, "metadata", None) or {})
    text = getattr(obs, "text", None) or ""
    parsed = parse_retro_pipe_text(text)
    ram_watch = dict(meta.get("ram_watch") or {})
    retro_game = str(meta.get("game") or parsed.get("game") or "")
    spec = TEMPORAL_GAME_SPECS.get(env_id)
    if spec is None and retro_game:
        for s in TEMPORAL_GAME_SPECS.values():
            if s.retro_game == retro_game:
                spec = s
                env_id = s.gym_env_id
                break

    available = list(meta.get("available_actions") or [])
    sim = {
        "frame_index": meta.get("frame_index"),
        "episode_reward": meta.get("episode_reward"),
        "step_reward": meta.get("step_reward"),
        "last_action": meta.get("last_action"),
        "action_history": list(meta.get("action_history") or []),
    }

    schema: dict[str, Any] = {
        "schema_kind": "gymv.temporal_visual_grounding",
        "schema_version": "1.0",
        "gym_env_id": env_id,
        "retro_integration": retro_game,
        "display_name": spec.display_name
        if spec
        else (retro_game.split("-")[0] if retro_game else "unknown"),
        "genre": spec.genre if spec else "unknown",
        "grounding_focus": spec.grounding_focus if spec else "",
        "visual": _image_size(obs),
        "simulation": sim,
        "parsed_text": dict(parsed),
        "ram_watch": {k: (v.item() if hasattr(v, "item") else v) for k, v in ram_watch.items()},
        "control": {"available_actions": available},
        "entities": _entities_from_retro(parsed, ram_watch, spec),
    }
    return schema


def visual_grounding_summary_line(schema: dict[str, Any]) -> str:
    """Single-line human/LLM hint (optional append to ``obs.text``)."""
    name = schema.get("display_name", "?")
    fi = (schema.get("simulation") or {}).get("frame_index")
    rw = schema.get("ram_watch") or {}
    keys = ",".join(sorted(rw.keys())[:6])
    return f"[visual_grounding {name} frame={fi} ram_keys={keys}]"


try:
    from gym_v.core import Env, Observation, ObservationWrapper

    class TemporalVisualGroundingWrapper(ObservationWrapper):
        """Attach ``metadata["visual_grounding"]`` (+ optional text suffix)."""

        _ENV_ID: str = ""

        def __init__(
            self,
            env: Env,
            *,
            append_text_summary: bool = False,
        ):
            super().__init__(env)
            self._append_text_summary = bool(append_text_summary)

        def _resolved_env_id(self) -> str:
            spec = getattr(self.env, "spec", None)
            if spec and getattr(spec, "id", None):
                return str(spec.id)
            cid = getattr(type(self), "_ENV_ID", "") or ""
            return str(cid) if cid else "Temporal/unknown-v0"

        def observation(self, observation: Observation) -> Observation:
            env_id = self._resolved_env_id()
            vg = build_temporal_visual_schema(env_id, observation)
            meta = dict(observation.metadata or {})
            meta["visual_grounding"] = vg
            text = observation.text
            if self._append_text_summary:
                line = visual_grounding_summary_line(vg)
                text = f"{text}\n{line}" if text else line
            return Observation(
                image=observation.image,
                text=text,
                metadata=meta,
            )

except ImportError:  # pragma: no cover
    TemporalVisualGroundingWrapper = None  # type: ignore[misc, assignment]


def _game_subclass(env_id: str, class_name: str) -> type:
    base = TemporalVisualGroundingWrapper
    if base is None:
        raise ImportError("gym_v is required for Temporal visual grounding wrappers")
    return type(class_name, (base,), {"_ENV_ID": env_id})


if TemporalVisualGroundingWrapper is not None:
    TemporalAirstrikerVisualGroundingWrapper = _game_subclass(
        "Temporal/Airstriker-v0", "TemporalAirstrikerVisualGroundingWrapper",
    )
    TemporalAlteredBeastVisualGroundingWrapper = _game_subclass(
        "Temporal/AlteredBeast-v0", "TemporalAlteredBeastVisualGroundingWrapper",
    )
    TemporalCastleOfIllusionVisualGroundingWrapper = _game_subclass(
        "Temporal/CastleOfIllusion-v0", "TemporalCastleOfIllusionVisualGroundingWrapper",
    )
    TemporalCastlevaniaBloodlinesVisualGroundingWrapper = _game_subclass(
        "Temporal/CastlevaniaBloodlines-v0",
        "TemporalCastlevaniaBloodlinesVisualGroundingWrapper",
    )
    TemporalColumnsVisualGroundingWrapper = _game_subclass(
        "Temporal/Columns-v0", "TemporalColumnsVisualGroundingWrapper",
    )
    TemporalDynamiteHeaddyVisualGroundingWrapper = _game_subclass(
        "Temporal/DynamiteHeaddy-v0", "TemporalDynamiteHeaddyVisualGroundingWrapper",
    )
    TemporalGoldenAxeVisualGroundingWrapper = _game_subclass(
        "Temporal/GoldenAxe-v0", "TemporalGoldenAxeVisualGroundingWrapper",
    )
    TemporalKidChameleonVisualGroundingWrapper = _game_subclass(
        "Temporal/KidChameleon-v0", "TemporalKidChameleonVisualGroundingWrapper",
    )
    TemporalMortalKombatIIVisualGroundingWrapper = _game_subclass(
        "Temporal/MortalKombatII-v0", "TemporalMortalKombatIIVisualGroundingWrapper",
    )
    TemporalSpaceHarrierIIVisualGroundingWrapper = _game_subclass(
        "Temporal/SpaceHarrierII-v0", "TemporalSpaceHarrierIIVisualGroundingWrapper",
    )
    TemporalStreetsOfRage2VisualGroundingWrapper = _game_subclass(
        "Temporal/StreetsOfRage2-v0", "TemporalStreetsOfRage2VisualGroundingWrapper",
    )
    TemporalStriderVisualGroundingWrapper = _game_subclass(
        "Temporal/Strider-v0", "TemporalStriderVisualGroundingWrapper",
    )
    TemporalThunderForceIIIVisualGroundingWrapper = _game_subclass(
        "Temporal/ThunderForceIII-v0", "TemporalThunderForceIIIVisualGroundingWrapper",
    )

    TEMPORAL_WRAPPER_BY_ENV_ID: dict[str, type] = {
        "Temporal/Airstriker-v0": TemporalAirstrikerVisualGroundingWrapper,
        "Temporal/AlteredBeast-v0": TemporalAlteredBeastVisualGroundingWrapper,
        "Temporal/CastleOfIllusion-v0": TemporalCastleOfIllusionVisualGroundingWrapper,
        "Temporal/CastlevaniaBloodlines-v0": TemporalCastlevaniaBloodlinesVisualGroundingWrapper,
        "Temporal/Columns-v0": TemporalColumnsVisualGroundingWrapper,
        "Temporal/DynamiteHeaddy-v0": TemporalDynamiteHeaddyVisualGroundingWrapper,
        "Temporal/GoldenAxe-v0": TemporalGoldenAxeVisualGroundingWrapper,
        "Temporal/KidChameleon-v0": TemporalKidChameleonVisualGroundingWrapper,
        "Temporal/MortalKombatII-v0": TemporalMortalKombatIIVisualGroundingWrapper,
        "Temporal/SpaceHarrierII-v0": TemporalSpaceHarrierIIVisualGroundingWrapper,
        "Temporal/StreetsOfRage2-v0": TemporalStreetsOfRage2VisualGroundingWrapper,
        "Temporal/Strider-v0": TemporalStriderVisualGroundingWrapper,
        "Temporal/ThunderForceIII-v0": TemporalThunderForceIIIVisualGroundingWrapper,
    }
else:  # pragma: no cover
    TemporalAirstrikerVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalAlteredBeastVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalCastleOfIllusionVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalCastlevaniaBloodlinesVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalColumnsVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalDynamiteHeaddyVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalGoldenAxeVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalKidChameleonVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalMortalKombatIIVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalSpaceHarrierIIVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalStreetsOfRage2VisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalStriderVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TemporalThunderForceIIIVisualGroundingWrapper = None  # type: ignore[misc, assignment]
    TEMPORAL_WRAPPER_BY_ENV_ID = {}
