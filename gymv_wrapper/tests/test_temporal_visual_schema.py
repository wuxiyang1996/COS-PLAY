"""Unit tests for Temporal visual grounding schema (no gym_v / ROM required)."""

from __future__ import annotations

from types import SimpleNamespace

from gymv_wrapper.temporal_visual_grounding import (
    TEMPORAL_GAME_SPECS,
    build_temporal_visual_schema,
    visual_grounding_summary_line,
)


def test_temporal_specs_cover_thirteen_games() -> None:
    assert len(TEMPORAL_GAME_SPECS) == 13
    assert "Temporal/Airstriker-v0" in TEMPORAL_GAME_SPECS


def test_build_schema_fuses_text_metadata_and_visual_size() -> None:
    obs = SimpleNamespace(
        text="Game: Airstriker-Genesis-v0 | Score: 10 | Lives: 3 | Frame: 5",
        metadata={
            "game": "Airstriker-Genesis-v0",
            "frame_index": 5,
            "episode_reward": 1.5,
            "step_reward": 0.1,
            "last_action": "RIGHT",
            "action_history": ["NOOP", "RIGHT"],
            "available_actions": ["A", "B", "RIGHT"],
            "ram_watch": {"custom_ram": 42},
        },
        image=SimpleNamespace(size=(320, 224), mode="RGB"),
    )
    schema = build_temporal_visual_schema("Temporal/Airstriker-v0", obs)
    assert schema["schema_kind"] == "gymv.temporal_visual_grounding"
    assert schema["gym_env_id"] == "Temporal/Airstriker-v0"
    assert schema["visual"]["width"] == 320
    assert schema["visual"]["height"] == 224
    assert schema["control"]["available_actions"] == ["A", "B", "RIGHT"]
    assert schema["ram_watch"]["custom_ram"] == 42
    labels = {e["label"] for e in schema["entities"]}
    assert "genesis_viewport" in labels
    assert "hud_score" in labels
    line = visual_grounding_summary_line(schema)
    assert "Airstriker" in line
    assert "custom_ram" in line


def test_resolve_spec_by_retro_game_when_env_id_unknown() -> None:
    obs = SimpleNamespace(
        text="Game: Columns-Genesis-v0 | Frame: 0",
        metadata={"game": "Columns-Genesis-v0"},
        image=None,
    )
    schema = build_temporal_visual_schema("unknown/Env-v0", obs)
    assert schema["gym_env_id"] == "Temporal/Columns-v0"
    assert schema["display_name"] == "Columns"
