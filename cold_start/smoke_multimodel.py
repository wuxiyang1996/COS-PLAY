"""smoke_multimodel.py — 60-second LLM-only smoke for the multi-provider
OSWorld actor pipeline.

Why this exists:
----------------
Every machine running ``run_osworld_multimodel.sh`` needs three things to
line up before the OSWorld VM is even worth booting:

  1. ``api_keys.py`` resolves so ``OPENROUTER_API_KEY`` (and/or
     ``OPENAI_API_KEY`` for the gpt5 provider) ends up in ``os.environ``.
  2. The driver's ``_build_client_and_route`` returns a usable client and
     the routed model id we expect (slash-prefixed for OpenRouter,
     bare for OpenAI direct).
  3. ``_chat_completion`` actually returns a tool-calling response with
     vision input — i.e. the model on OpenRouter accepts our exact
     ``tools`` + ``tool_choice`` + ``image_url`` shape.

A live OSWorld run takes 30+ minutes per task to fail on a credentials /
plumbing bug; this smoke validates the same code paths in ~10 seconds per
provider. Run it FIRST when bringing up a new machine, then launch the
real eval.

Usage:
    python cold_start/smoke_multimodel.py
    python cold_start/smoke_multimodel.py --provider claude-sonnet
    python cold_start/smoke_multimodel.py --provider gemini-pro
    python cold_start/smoke_multimodel.py --provider qwen3-vl
    python cold_start/smoke_multimodel.py --provider gpt5
    python cold_start/smoke_multimodel.py --all          # default
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

# Make the codebase importable when launched from anywhere
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))  # codebase root
sys.path.insert(0, str(_HERE.parent.parent.parent))  # workspace root


# Mirror the wrapper's provider table (kept in sync by hand — there are
# only 7 entries and we want this module to stand alone without a JSON
# manifest).
PROVIDER_MODEL_TABLE: List[Tuple[str, str, str]] = [
    ("claude-sonnet",  "anthropic/claude-sonnet-4.6",       "openrouter"),
    ("claude-opus",    "anthropic/claude-opus-4.7",         "openrouter"),
    ("gemini-pro",     "google/gemini-2.5-pro",             "openrouter"),
    ("gemini-3-pro",   "google/gemini-3.1-pro-preview",     "openrouter"),
    ("qwen3-vl",       "qwen/qwen3-vl-235b-a22b-instruct",  "openrouter"),
    ("gpt5",           "gpt-5.4",                           "openai-direct"),
    ("gpt5-or",        "openai/gpt-5.4",                    "openrouter"),
]

PROVIDER_BY_NAME: Dict[str, Tuple[str, str]] = {
    name: (model_id, route) for name, model_id, route in PROVIDER_MODEL_TABLE
}


def _build_test_image_data_url() -> str:
    """Tiny 320x200 PNG with a labelled rectangle.

    Empty-/red-only images sometimes trip Gemini's safety classifier.
    A labelled grayscale rectangle reliably passes safety while still
    being non-trivial enough that the model doesn't hallucinate.
    """
    from PIL import Image, ImageDraw  # imported lazily so missing PIL
    # produces a friendly message rather than an ImportError on probe
    img = Image.new("RGB", (320, 200), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 110, 50], outline="black", width=2)
    d.text((20, 20), "File Menu", fill="black")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _action_tool_schema() -> Dict:
    """A reduced-but-real ``choose_action`` schema mirroring what
    ``_build_action_tools`` ships in the driver (subgoal + reasoning +
    action_string). We keep it identical so any breakage caught here
    will reproduce in the live actor."""
    return {
        "type": "function",
        "function": {
            "name": "choose_action",
            "description": "Pick the next pyautogui-style action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subgoal": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "action_string": {"type": "string"},
                },
                "required": ["subgoal", "reasoning", "action_string"],
            },
        },
    }


def _smoke_one(
    provider: str,
    *,
    verbose: bool = False,
    reasoning_effort: str = "low",
) -> Tuple[bool, str]:
    """Run a 1-call action smoke + a 1-call schema-VLM smoke.

    Returns ``(ok, message)``. ``ok`` is True only when both calls
    succeed. The message names whichever call failed so the operator can
    see which axis broke (auth / vision / tool-calling / routing).
    """
    if provider not in PROVIDER_BY_NAME:
        return False, f"unknown provider {provider!r}"
    model_id, route = PROVIDER_BY_NAME[provider]

    # Lazy import of the driver (heavy — pulls in osworld + pillow + …).
    try:
        from cold_start.generate_cold_start_actor_osworld import (
            _build_client_and_route, _chat_completion, _is_reasoning_model,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"driver import failed: {exc!r}"

    data_url = _build_test_image_data_url()
    tool = _action_tool_schema()
    tool_choice = {"type": "function", "function": {"name": "choose_action"}}

    # --- Pass 1: full action call shape (vision + tools + tool_choice) ---
    client, routed = _build_client_and_route(model=model_id)
    if client is None:
        return False, (
            f"client build returned None — likely missing API key for "
            f"route={route}. Check api_keys.py / env."
        )
    if verbose:
        print(f"  routed_model={routed}  reasoning_model={_is_reasoning_model(routed)}")

    msgs_action = [
        {"role": "system",
         "content": "You are an OSWorld actor. Use the choose_action tool."},
        {"role": "user",
         "content": [
             {"type": "text",
              "text": (
                  "Look at the screenshot. Then call choose_action with "
                  "subgoal=\"open File menu\", a one-sentence reasoning, "
                  "and action_string=\"pyautogui.click(60, 30)\"."
              )},
             {"type": "image_url", "image_url": {"url": data_url}},
         ]},
    ]

    t0 = time.time()
    try:
        resp = _chat_completion(
            client, model=routed, messages=msgs_action,
            temperature=0.0, max_tokens=600,
            tools=[tool], tool_choice=tool_choice,
            # Default ``low`` matches the production tier picked by
            # ``run_osworld_multimodel.sh``. The driver drops the
            # parameter for non-OpenAI-reasoning models, so a passing
            # smoke proves the drop is silent.
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"action-LLM call failed: {exc!r}"
    dt_action = time.time() - t0

    ch = resp.choices[0]
    tcs = ch.message.tool_calls or []
    if not tcs:
        return False, (
            f"action call returned no tool_calls (finish_reason={ch.finish_reason}, "
            f"text={(ch.message.content or '')[:120]!r})"
        )
    args = tcs[0].function.arguments or ""
    try:
        parsed = json.loads(args)
    except Exception:
        return False, f"action call returned non-JSON tool args: {args[:120]!r}"
    missing = [k for k in ("subgoal", "reasoning", "action_string") if k not in parsed]
    if missing:
        return False, f"action tool args missing fields {missing}: {args[:120]!r}"

    if verbose:
        print(f"  action OK ({dt_action:.1f}s)  args={args[:100]}")

    # --- Pass 2: schema-VLM call shape (vision, no tools, max budget) ---
    msgs_schema = [
        {"role": "system",
         "content": "Return a 1-line JSON description of the screenshot: {\"summary\": \"...\"}."},
        {"role": "user",
         "content": [
             {"type": "text", "text": "Describe this screenshot."},
             {"type": "image_url", "image_url": {"url": data_url}},
         ]},
    ]
    t0 = time.time()
    try:
        resp = _chat_completion(
            client, model=routed, messages=msgs_schema,
            temperature=0.2, max_tokens=300,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"schema-VLM call failed: {exc!r}"
    dt_schema = time.time() - t0
    txt = (resp.choices[0].message.content or "").strip()
    if not txt:
        return False, (
            f"schema-VLM call returned empty content "
            f"(finish_reason={resp.choices[0].finish_reason})"
        )
    if verbose:
        print(f"  schema OK ({dt_schema:.1f}s)  text={txt[:100]!r}")
    return True, (
        f"OK   action={dt_action:.1f}s schema={dt_schema:.1f}s "
        f"action_args={args[:80]}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=[n for n, _, _ in PROVIDER_MODEL_TABLE],
                   help="single provider to smoke. omit for --all")
    p.add_argument("--all", action="store_true",
                   help="run every provider in the table (default if no --provider)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--skip", default="",
                   help="comma-separated list of providers to skip when --all")
    p.add_argument(
        "--reasoning-effort", "--reasoning_effort",
        default="low",
        choices=["minimal", "low", "medium", "high"],
        help=(
            "reasoning_effort to forward to the action + schema LLM "
            "calls. Default ``low`` matches the production tier picked "
            "by run_osworld_multimodel.sh. ``minimal`` is OpenRouter-"
            "only — OpenAI direct gpt-5.x rejects it (HTTP 400)."
        ),
    )
    args = p.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if args.provider:
        targets = [args.provider]
    else:
        targets = [n for n, _, _ in PROVIDER_MODEL_TABLE if n not in skip]

    print("=" * 64)
    print(" multimodel smoke — LLM-only, no OSWorld VM needed")
    print("=" * 64)
    rows: List[Tuple[str, bool, str]] = []
    for provider in targets:
        model_id, route = PROVIDER_BY_NAME[provider]
        print(f"\n[{provider:<14}] {model_id}  ({route})  reasoning={args.reasoning_effort}")
        ok, msg = _smoke_one(
            provider,
            verbose=args.verbose,
            reasoning_effort=args.reasoning_effort,
        )
        rows.append((provider, ok, msg))
        print(f"  {'PASS' if ok else 'FAIL'}: {msg}")

    print("\n" + "=" * 64)
    print(" Summary")
    print("=" * 64)
    n_ok = sum(1 for _, ok, _ in rows if ok)
    for provider, ok, msg in rows:
        marker = "PASS" if ok else "FAIL"
        print(f"  {marker:<4}  {provider:<14}  {msg}")
    print(f"\n  total: {n_ok}/{len(rows)} passed")

    if n_ok < len(rows):
        print("\n  Hint: failing providers usually mean either:")
        print("    - missing OPENROUTER_API_KEY (or openai_api_key for gpt5)")
        print("    - the OpenRouter org has not enabled that model family")
        print("    - the provider's id has changed (rerun")
        print("      `python -m openrouter list` or browse openrouter.ai/models)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
