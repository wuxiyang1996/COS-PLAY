"""Cross-domain ``cascaded_ground`` smoke test.

Each per-domain adapter contract test now lives next to its wrapper:

* :mod:`gymv_wrapper.tests.test_gpt4o_parsers`             — Gym-V
* :mod:`browsergym_wrapper.tests.test_gpt4o_parsers`       — BrowserGym
* :mod:`osworld_wrapper.tests.test_gpt4o_parsers`          — OSWorld (desktop)
* :mod:`visual_reasoning_wrapper.tests.test_gpt4o_parsers` — TIR-Bench, Video-Holmes, synthesizers

What stays in ``vlm_wrapper/tests`` is the ONE end-to-end test that
exercises the cross-domain :func:`vlm_wrapper.ground.cascaded_ground`
driver itself — multi-hop tool-calling, head escalation, the
``<evidence>`` block — which is genuinely owned by the cross-domain
core.

Run offline (the test below is ``@pytest.mark.live`` so it is skipped
unless an API key is present)::

    pytest vlm_wrapper/tests/test_gpt4o_parsers.py -q

Run including the live GPT-4o call::

    pytest -m "live or not live" vlm_wrapper/tests/test_gpt4o_parsers.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMAS_DIR = REPO_ROOT / "out" / "schemas"
API_KEY = os.environ.get("VLM_TEST_API_KEY") or os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("VLM_LABEL_MODEL", "gpt-4o")

live = pytest.mark.live
needs_api = pytest.mark.skipif(
    not API_KEY, reason="OPENAI_API_KEY / VLM_TEST_API_KEY not set"
)


def _save_schema(case: str, schema: str) -> None:
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    (SCHEMAS_DIR / f"{case}.schema.txt").write_text(schema, encoding="utf-8")


def _assert_valid_schema(schema: str | None, case: str) -> None:
    assert schema, f"{case}: parser returned no schema"
    assert "<state>" in schema and "</state>" in schema, (
        f"{case}: missing <state>…</state>:\n{schema[:400]}"
    )
    has_entities = "<entities>" in schema
    has_events = "<events>" in schema
    has_answer = "<answer>" in schema
    assert has_entities or has_events or has_answer, (
        f"{case}: schema lacks any of <entities>/<events>/<answer>:\n"
        f"{schema[:400]}"
    )


@live
@needs_api
def test_live_gymv_tool_loop_schema() -> None:
    """Gym-V via VLM + tool-calling (cascaded_ground, chain=['tool_loop']).

    Exercises the full multi-hop pipeline for an interactive env: the
    VLM sees the rendered 2048 board AND has access to the gymv tool
    registry (list_entities, query_entity_pos, check_relation,
    count_merge_candidates, get_state_flags, list_valid_actions, …).
    We provide obs_text alongside the image so the tool handlers can
    return ground-truth data — the point of this test is to prove the
    VLM actually uses tools to ground the schema, not to hide them.

    Asserts (in order):
      1. a parseable <state> schema comes back;
      2. cascaded_ground picks the tool_loop head;
      3. the semantic validator marks the schema valid;
      4. GPT-4o issued at least one tool call (tool_trace non-empty);
      5. the schema's <actions> block copies the env's valid actions
         verbatim (no "slide_left"-style hallucinations);
      6. an <evidence> block records the reasoning hops so downstream
         skill learning can cite them.
    """
    from vlm_wrapper.ground import GroundingRequest, cascaded_ground

    # Hand-drawn mid-game 2048 board (richer than the early-game
    # synthesizer) so GPT-4o has several non-empty tiles to reason
    # about.  Keeps the test above the gymv entity-count floor (>=3)
    # without leaning on the `_synthesize_2048_board` default.
    board = [
        [2, 4, 0, 0],
        [0, 2, 4, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]
    tile_colors = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
    }
    image = Image.new("RGB", (480, 480), (187, 173, 160))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except Exception:
        font = ImageFont.load_default()
    cell, pad = 110, 10
    for r in range(4):
        for c in range(4):
            x0 = pad + c * (cell + pad)
            y0 = pad + r * (cell + pad)
            val = board[r][c]
            draw.rounded_rectangle(
                [x0, y0, x0 + cell, y0 + cell], radius=6,
                fill=tile_colors.get(val, (237, 224, 200)),
            )
            if val:
                draw.text(
                    (x0 + cell / 2, y0 + cell / 2), str(val),
                    fill=(119, 110, 101), anchor="mm", font=font,
                )
    obs_text = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in board
    )
    game_rules = (
        "You are playing 2048.  Slide tiles in one of the four "
        "directions; tiles with the same value merge.  Valid moves: "
        "[Up], [Down], [Left], [Right]."
    )
    valid_actions = ["[Up]", "[Down]", "[Left]", "[Right]"]

    req = GroundingRequest(
        images=image,
        goal="Reach 2048",
        domain="gymv",
        output_mode="actions",
        task_id="Game2048-v0",
        step=0,
        context={
            "description": game_rules,
            "obs_text": obs_text,             # wired into tool handlers
            "valid_actions": valid_actions,
            # Hide obs_text from the VLM prompt so GPT-4o has to call
            # tools (list_entities / get_grid_state / query_entity_pos
            # / check_relation / count_merge_candidates) to ground
            # its schema — otherwise it would just paraphrase the grid.
            "show_obs_text": False,
        },
        max_entities=12,
        max_rounds=6,
        model=MODEL,
        api_key=API_KEY,
    )
    result = cascaded_ground(
        req,
        image_size=image.size,
        chain=["tool_loop"],
    )

    schema = result.schema
    _assert_valid_schema(schema, "gymv_tool_loop")
    _save_schema("gymv_tool_loop", schema)

    assert result.head_used == "tool_loop", (
        f"expected tool_loop head, got {result.head_used!r}"
    )

    val = result.validation
    assert val is not None, "cascaded_ground should attach a ValidationResult"
    assert val.valid, (
        f"tool_loop schema failed semantic validation: "
        f"errors={val.errors} missing_slots={val.missing_slots}"
    )

    assert result.tool_trace, (
        "GPT-4o did not invoke any tools — tool_loop should produce "
        "at least one call in the trace"
    )
    gymv_tool_names = {
        "query_entity_pos", "list_entities", "check_relation",
        "get_state_flags", "list_valid_actions", "get_grid_state",
        "check_deadlock", "spatial_analysis", "count_merge_candidates",
    }
    visual_tool_names = {
        "detect_objects", "grounded_detect", "describe_region",
        "zoom_region", "visual_search", "count_objects",
        "classify_scene", "spatial_query", "measure_distance",
        "extract_colors", "read_text_region",
    }
    known_tools = gymv_tool_names | visual_tool_names
    called_tools = {
        (tc.get("call", {}) or {}).get("name", "") for tc in result.tool_trace
    }
    assert called_tools & known_tools, (
        f"tool_trace only contains unknown tool names {called_tools} — "
        f"expected at least one call to a gymv or visual tool"
    )

    for action in valid_actions:
        if action in schema:
            break
    else:  # pragma: no cover — defensive
        raise AssertionError(
            f"none of the valid actions {valid_actions} appear in the "
            f"<actions> block of the schema:\n{schema[-400:]}"
        )
    assert "slide_" not in schema.lower(), (
        "schema contains invented 'slide_*' actions — the VLM ignored "
        "the valid-actions instruction"
    )

    assert "<evidence>" in schema, (
        "tool_loop schema should record reasoning hops in <evidence>"
    )
