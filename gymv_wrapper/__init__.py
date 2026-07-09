"""Gym-V–specific visual grounding, heuristics, VLM adapters, and tools.

Code that only concerns Gym-V / stable-retro lives here; cross-domain schema
and prompts remain in ``vlm_wrapper.schema``.

**Temporal / stable-retro (13 Genesis games)**
Use :class:`gymv_wrapper.temporal_visual_grounding.TemporalVisualGroundingWrapper`
or a per-game subclass (e.g. :class:`TemporalAirstrikerVisualGroundingWrapper`)
to attach a JSON-serializable ``visual_grounding`` dict (and summary text) to
each :class:`gym_v.Observation` based on the frame, RAM watch, and env text.

**VLM + heuristics + tools**
- :func:`gymv_wrapper.adapter.generate_label` — screenshot → vision LLM → ``<state>`` schema
- :class:`gymv_wrapper.adapter.GymVSchemaWrapper` — online wrapper (optional ``gym_v``)
- :func:`gymv_wrapper.heuristic.text_to_schema` — text → schema heuristics
- :func:`gymv_wrapper.tools.build_gymv_registry` — grounding tools for tool-loop
"""

from __future__ import annotations

from gymv_wrapper.adapter import GymVSchemaWrapper, generate_label
from gymv_wrapper.heuristic import text_to_schema
from gymv_wrapper.tools import build_gymv_registry

__all__ = [
    "generate_label",
    "text_to_schema",
    "build_gymv_registry",
    "build_temporal_visual_schema",
    "GymVSchemaWrapper",
]

from gymv_wrapper.temporal_visual_grounding import (
    TEMPORAL_GAME_SPECS,
    TEMPORAL_WRAPPER_BY_ENV_ID,
    TemporalAirstrikerVisualGroundingWrapper,
    TemporalAlteredBeastVisualGroundingWrapper,
    TemporalCastleOfIllusionVisualGroundingWrapper,
    TemporalCastlevaniaBloodlinesVisualGroundingWrapper,
    TemporalColumnsVisualGroundingWrapper,
    TemporalDynamiteHeaddyVisualGroundingWrapper,
    TemporalGoldenAxeVisualGroundingWrapper,
    TemporalKidChameleonVisualGroundingWrapper,
    TemporalMortalKombatIIVisualGroundingWrapper,
    TemporalSpaceHarrierIIVisualGroundingWrapper,
    TemporalStreetsOfRage2VisualGroundingWrapper,
    TemporalStriderVisualGroundingWrapper,
    TemporalThunderForceIIIVisualGroundingWrapper,
    TemporalVisualGroundingWrapper,
    build_temporal_visual_schema,
)

__all__ += [
    "TEMPORAL_GAME_SPECS",
    "TEMPORAL_WRAPPER_BY_ENV_ID",
    "TemporalVisualGroundingWrapper",
    "build_temporal_visual_schema",
    "TemporalAirstrikerVisualGroundingWrapper",
    "TemporalAlteredBeastVisualGroundingWrapper",
    "TemporalCastleOfIllusionVisualGroundingWrapper",
    "TemporalCastlevaniaBloodlinesVisualGroundingWrapper",
    "TemporalColumnsVisualGroundingWrapper",
    "TemporalDynamiteHeaddyVisualGroundingWrapper",
    "TemporalGoldenAxeVisualGroundingWrapper",
    "TemporalKidChameleonVisualGroundingWrapper",
    "TemporalMortalKombatIIVisualGroundingWrapper",
    "TemporalSpaceHarrierIIVisualGroundingWrapper",
    "TemporalStreetsOfRage2VisualGroundingWrapper",
    "TemporalStriderVisualGroundingWrapper",
    "TemporalThunderForceIIIVisualGroundingWrapper",
]
