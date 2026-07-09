#!/usr/bin/env python3
"""Demo: Visual grounding models → structured <state> schema (BrowserGym).

Shows the full pipeline discussed in vlm_wrapper/PLAN_GROUNDING.md applied
to a BrowserGym page:

  Mode A – **Direct grounding** (no API needed)
      Screenshot → OmniParser-v2 (YOLO + OCR + Florence-2) → schema
      Falls back to OCR-only if OmniParser weights aren't downloaded yet.

  Mode B – **VLM + visual tools** (needs API key)
      Screenshot → GPT-4o/4.1 sees the image and calls tool functions
      (detect_objects, spatial_query, visual_search, describe_region …)
      backed by local vision models → produces final schema.
      This is the *multi-hop visual reasoning* loop: the VLM gathers
      ground-truth evidence from specialised models before committing
      to a structured output.

  Mode C – **All three heads compared** (needs API key)
      Runs Head 1 (heuristic/AXTree), Head 3 (grounding), and
      the VLM tool loop side-by-side on the same screenshot.

Usage:
    # Mode A — direct grounding only (no API key)
    python -m browsergym_wrapper.demo_visual_grounding

    # Mode B — VLM tool-calling loop (needs API key)
    python -m browsergym_wrapper.demo_visual_grounding --vlm-tools \\
        --api-key sk-... --model openai/gpt-4.1

    # Mode C — full comparison
    python -m browsergym_wrapper.demo_visual_grounding --compare-all \\
        --api-key sk-... --model openai/gpt-4.1

    # Custom screenshot
    python -m browsergym_wrapper.demo_visual_grounding --image screenshot.png

    # Save annotated image showing detected elements
    python -m browsergym_wrapper.demo_visual_grounding --save-annotated out.png

    # Use OpenRouter
    python -m browsergym_wrapper.demo_visual_grounding --vlm-tools \\
        --api-key $OPENROUTER_KEY \\
        --base-url https://openrouter.ai/api/v1 \\
        --model openai/gpt-4.1
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Shared helpers ────────────────────────────────────────────────────

def _load_or_generate_image(image_path: str | None) -> Image.Image:
    """Load a user-supplied screenshot or generate the synthetic shopping page."""
    if image_path:
        img = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {image_path} ({img.size[0]}x{img.size[1]})")
        return img

    saved = Path(__file__).parent / "example_screenshot.png"
    if saved.exists():
        img = Image.open(saved).convert("RGB")
        print(f"Using cached screenshot: {saved} ({img.size[0]}x{img.size[1]})")
        return img

    from browsergym_wrapper.example import make_fake_screenshot
    arr = make_fake_screenshot()
    img = Image.fromarray(arr)
    print(f"Generated synthetic shopping page ({img.size[0]}x{img.size[1]})")
    return img


def _sep(title: str, char: str = "=", width: int = 70) -> str:
    line = char * width
    return f"\n{line}\n  {title}\n{line}"


# ── Mode A: Direct grounding (OmniParser-v2 or OCR fallback) ─────────

def run_direct_grounding(image: Image.Image, args) -> dict:
    """Run the local grounding pipeline and produce a schema."""

    print(_sep("MODE A: Direct Visual Grounding (local, no API)"))

    t0 = time.time()
    elements, schema, warnings = None, None, None

    # Try OmniParser-v2 first, fall back to OCR-only
    try:
        from browsergym_wrapper.grounding import grounding_image_to_schema
        result = grounding_image_to_schema(
            image,
            goal=args.goal,
            task_id="demo.visual_grounding",
            step=1,
            url="https://shopmart.example.com/search?q=red+jacket",
            max_entities=args.max_entities,
        )
        elements = result["elements"]
        schema = result["schema"]
        warnings = result["warnings"]
        backend = "OmniParser-v2 (YOLO + OCR + Florence-2)"
    except (ImportError, ModuleNotFoundError) as exc:
        backend = f"OCR-only fallback ({exc.__class__.__name__}: {exc})"
    except Exception as exc:
        backend = f"OCR-only fallback (OmniParser error: {exc})"

    if elements is None:
        elements, schema, warnings = _ocr_fallback_grounding(image, args)

    print(f"  Backend: {backend}")

    elapsed = time.time() - t0

    print(f"\n  Detected {len(elements)} UI elements in {elapsed:.2f}s:")
    for i, el in enumerate(elements[:20]):
        if hasattr(el, "bbox"):
            bbox = el.bbox
            print(f"    [{i:2d}] {el.element_type:5s} | {el.label:40s} | "
                  f"({bbox.x},{bbox.y},{bbox.w},{bbox.h}) | "
                  f"{'clickable' if el.interactable else 'static':>9s} | {el.source}")
        else:
            print(f"    [{i:2d}] {el}")

    print(f"\n  Schema ({len(schema)} chars):")
    print(textwrap.indent(schema, "    "))

    if warnings:
        print(f"\n  Warnings: {warnings}")
    else:
        print("\n  Validation: PASSED")

    return {"elements": elements, "schema": schema, "warnings": warnings, "elapsed": elapsed}


def _ocr_fallback_grounding(image: Image.Image, args):
    """Minimal grounding using only EasyOCR when OmniParser isn't available."""
    try:
        import easyocr
    except ImportError:
        print("\n  [!] Neither OmniParser nor EasyOCR available.")
        print("      Install: pip install easyocr")
        print("      Or for full pipeline: pip install ultralytics transformers easyocr")
        return [], "<state>\n(no detection backend available)\n</state>", ["no backend"]

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    raw = reader.readtext(np.array(image))

    from dataclasses import dataclass

    @dataclass
    class _SimpleElement:
        element_type: str
        label: str
        bbox: object
        interactable: bool
        confidence: float
        source: str

    @dataclass
    class _SimpleBBox:
        x: int; y: int; w: int; h: int

    elements = []
    for bbox_pts, text, conf in raw:
        flat = [int(p) for pt in bbox_pts for p in pt]
        x_min, y_min = min(flat[::2]), min(flat[1::2])
        x_max, y_max = max(flat[::2]), max(flat[1::2])
        elements.append(_SimpleElement(
            element_type="text",
            label=text,
            bbox=_SimpleBBox(x_min, y_min, x_max - x_min, y_max - y_min),
            interactable=False,
            confidence=conf,
            source="easyocr",
        ))

    lines = ["<state>", "domain=browser", "task=demo.visual_grounding",
             f"goal={args.goal}", "step=1", "", "<entities>"]
    for i, el in enumerate(elements[:args.max_entities]):
        eid = f"e{i+1}"
        pos = f"{el.bbox.x},{el.bbox.y},{el.bbox.w},{el.bbox.h}"
        lines.append(f"{eid}[type=text, label='{el.label}', pos={pos}]")
    lines += ["", "<attributes>"]
    for i in range(min(len(elements), args.max_entities)):
        lines.append(f"e{i+1}.state=visible")
    lines += ["", "<relations>", "", "<state_flags>", "progress=null", "phase=null",
              "error=null", "dialog_open=false", "grounding_model=easyocr-fallback",
              "", "<targets>", "target=null", "", "</state>"]

    schema = "\n".join(lines)
    from vlm_wrapper.schema import validate_schema
    warnings = validate_schema(schema)
    return elements, schema, warnings


# ── Mode B: VLM + Visual Tools (multi-hop reasoning) ─────────────────

def run_vlm_tool_loop(image: Image.Image, args) -> dict:
    """Run the VLM tool-calling loop with visual grounding tools."""

    print(_sep("MODE B: VLM + Visual Grounding Tools (multi-hop reasoning)"))

    if not args.api_key:
        print("\n  [!] API key required for VLM tool loop.")
        print("      Use: --api-key sk-... or set OPENAI_API_KEY")
        return {}

    from visual_reasoning_wrapper.tools_visual import build_visual_registry
    from vlm_wrapper.tool_loop import run_tool_loop

    registry = build_visual_registry(image)
    tool_names = registry.tool_names()
    print(f"  Model: {args.model}")
    print(f"  Tools available ({len(tool_names)}): {', '.join(tool_names)}")
    print(f"  Max rounds: {args.max_rounds}")
    print()

    t0 = time.time()
    result = run_tool_loop(
        image,
        domain="browser",
        registry=registry,
        goal=args.goal,
        task_id="demo.visual_grounding",
        step=1,
        extra_context=(
            "You have vision-model tools for precise element detection and "
            "spatial reasoning. Use detect_objects first to get ground-truth "
            "element positions, then spatial_query or visual_search as needed."
        ),
        max_entities=args.max_entities,
        max_rounds=args.max_rounds,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    elapsed = time.time() - t0

    print(f"  Completed in {result['rounds']} round(s), {elapsed:.2f}s")

    if result["tool_trace"]:
        print(f"\n  Tool calls ({len(result['tool_trace'])}):")
        for i, tc in enumerate(result["tool_trace"]):
            call = tc["call"]
            res = tc["result"]
            args_str = json.dumps(call["arguments"], default=str)
            if len(args_str) > 80:
                args_str = args_str[:77] + "..."
            res_str = json.dumps(res, default=str)
            if len(res_str) > 120:
                res_str = res_str[:117] + "..."
            print(f"    [{i+1}] {call['name']}({args_str})")
            print(f"         → {res_str}")

    if result["schema"]:
        print(f"\n  Schema ({len(result['schema'])} chars):")
        print(textwrap.indent(result["schema"], "    "))
    else:
        print("\n  [NO SCHEMA PARSED]")
        if result["raw"]:
            print("  Raw output (first 500 chars):")
            print(textwrap.indent(result["raw"][:500], "    "))

    if result["warnings"]:
        print(f"\n  Warnings: {result['warnings']}")
    else:
        print("\n  Validation: PASSED")

    return {**result, "elapsed": elapsed}


# ── Mode C: Full comparison (Head 1 + Head 3 + VLM tools) ────────────

def run_comparison(image: Image.Image, args):
    """Compare all available heads on the same screenshot."""

    print(_sep("MODE C: Full Comparison", char="*"))

    results = {}

    # Head 1: Heuristic (needs AXTree — use fake obs)
    print(_sep("Head 1: Heuristic (AXTree → schema)", char="-"))
    try:
        from browsergym_wrapper.example import (
            make_fake_axtree, make_fake_extra_props,
        )
        from browsergym_wrapper.heuristic import obs_to_schema

        obs = {
            "axtree_object": make_fake_axtree(),
            "extra_element_properties": make_fake_extra_props(),
            "focused_element_bid": "a2",
            "goal": args.goal,
            "url": "https://shopmart.example.com/search?q=red+jacket",
            "last_action_error": "",
            "open_pages_urls": ["https://shopmart.example.com/search?q=red+jacket"],
        }
        t0 = time.time()
        h1_schema = obs_to_schema(obs, step=1, task_id="demo.comparison.heuristic")
        h1_elapsed = time.time() - t0

        print(f"  Elapsed: {h1_elapsed * 1000:.1f}ms")
        print(textwrap.indent(h1_schema, "    "))
        results["head1"] = {"schema": h1_schema, "elapsed": h1_elapsed}
    except Exception as e:
        print(f"  Head 1 failed: {e}")
        results["head1"] = None

    # Head 3: Direct grounding
    h3_result = run_direct_grounding(image, args)
    results["head3"] = h3_result

    # VLM tool loop
    if args.api_key:
        vlm_result = run_vlm_tool_loop(image, args)
        results["vlm_tools"] = vlm_result
    else:
        print(f"\n  Skipping VLM tool loop (no --api-key).")
        results["vlm_tools"] = None

    # Summary
    print(_sep("COMPARISON SUMMARY", char="*"))
    from vlm_wrapper.schema import count_entities

    for label, key in [("Head 1 (heuristic)", "head1"),
                       ("Head 3 (grounding)", "head3"),
                       ("VLM + tools", "vlm_tools")]:
        r = results.get(key)
        if r and r.get("schema"):
            n = count_entities(r["schema"])
            t = r.get("elapsed", 0)
            print(f"  {label:25s}: {n:2d} entities | {t:.2f}s | "
                  f"schema={len(r['schema'])} chars")
        elif r is None:
            print(f"  {label:25s}: skipped")
        else:
            print(f"  {label:25s}: no schema produced")


# ── Annotated image ───────────────────────────────────────────────────

def save_annotated(image: Image.Image, path: str):
    """Save an image with detected elements overlaid."""
    try:
        from vlm_wrapper.grounding import parse_screen_annotated
        _, annotated = parse_screen_annotated(image)
        annotated.save(path)
        print(f"\n  Annotated image saved: {path}")
    except ImportError:
        from visual_reasoning_wrapper.tools_visual import _VisualState
        state = _VisualState(image)
        dets = state.detect()
        from PIL import ImageDraw
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        for i, d in enumerate(dets):
            x, y, w, h = d.bbox
            color = (200, 0, 0) if d.element_type == "icon" else (0, 200, 0)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            draw.text((x + 2, max(0, y - 12)), f"{i}: {d.label[:20]}", fill=color)
        draw_img.save(path)
        print(f"\n  Annotated image saved (OCR-only): {path}")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Demo: visual grounding models → structured schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Direct grounding only (no API key needed)
              python -m browsergym_wrapper.demo_visual_grounding

              # VLM tool loop (multi-hop visual reasoning)
              python -m browsergym_wrapper.demo_visual_grounding --vlm-tools --api-key sk-...

              # Compare all approaches
              python -m browsergym_wrapper.demo_visual_grounding --compare-all --api-key sk-...
        """),
    )
    parser.add_argument("--image", default=None, help="Path to screenshot PNG")
    parser.add_argument("--goal", default="Find the cheapest red jacket and add to cart",
                        help="Task goal for schema context")
    parser.add_argument("--max-entities", type=int, default=25,
                        help="Maximum entities in schema")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--vlm-tools", action="store_true",
                      help="Run VLM tool-calling loop (Mode B)")
    mode.add_argument("--compare-all", action="store_true",
                      help="Compare all heads (Mode C)")

    parser.add_argument("--api-key", default=None,
                        help="OpenAI or OpenRouter API key")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--model", default="gpt-4o",
                        help="VLM model name (default: gpt-4o)")
    parser.add_argument("--max-rounds", type=int, default=5,
                        help="Max tool-calling rounds")

    parser.add_argument("--save-annotated", default=None,
                        help="Save annotated image with detected elements")
    parser.add_argument("--save-screenshot", action="store_true",
                        help="Save the input screenshot to disk")
    args = parser.parse_args()

    print("=" * 70)
    print("  Visual Grounding -> Structured Output Demo")
    print("  Pipeline: screenshot -> grounding model -> <state> schema")
    print("=" * 70)

    image = _load_or_generate_image(args.image)

    if args.save_screenshot:
        out = Path(__file__).parent / "demo_screenshot.png"
        image.save(str(out))
        print(f"  Screenshot saved: {out}")

    if args.compare_all:
        run_comparison(image, args)
    elif args.vlm_tools:
        run_vlm_tool_loop(image, args)
    else:
        run_direct_grounding(image, args)

    if args.save_annotated:
        save_annotated(image, args.save_annotated)

    print("\nDone.")


if __name__ == "__main__":
    main()
