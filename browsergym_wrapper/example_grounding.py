#!/usr/bin/env python3
"""Example: OmniParser-v2 grounding on a BrowserGym screenshot (Head 3).

Demonstrates the grounding pipeline:
  1. Load or generate a screenshot (reuses the synthetic shopping page)
  2. Run OmniParser-v2 to detect UI elements (YOLO + OCR + Florence-2)
  3. Map detections into the canonical <state>…</state> schema
  4. Optionally compare with Head 1 (heuristic) output

Usage:
    # Head 3 only — requires OmniParser deps (ultralytics, easyocr, transformers)
    python -m browsergym_wrapper.example_grounding

    # Compare Head 1 (heuristic) vs Head 3 (grounding)
    python -m browsergym_wrapper.example_grounding --compare

    # Use a custom screenshot
    python -m browsergym_wrapper.example_grounding --image path/to/screenshot.png

    # Save the annotated image
    python -m browsergym_wrapper.example_grounding --save-annotated annotated.png

    # Use PaddleOCR instead of EasyOCR
    python -m browsergym_wrapper.example_grounding --paddleocr

    # Skip Florence-2 captioning (faster, labels will be "unknown" for icons)
    python -m browsergym_wrapper.example_grounding --no-caption
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_or_generate_image(image_path: str | None) -> Image.Image:
    """Load a screenshot or generate the synthetic shopping page."""
    if image_path:
        return Image.open(image_path).convert("RGB")

    example_png = Path(__file__).parent / "example_screenshot.png"
    if example_png.exists():
        return Image.open(example_png).convert("RGB")

    from browsergym_wrapper.example import make_fake_screenshot
    arr = make_fake_screenshot()
    return Image.fromarray(arr)


def run_head3(image: Image.Image, args) -> dict:
    """Run Head 3 (OmniParser-v2 grounding)."""
    from browsergym_wrapper.grounding import grounding_image_to_schema

    result = grounding_image_to_schema(
        image,
        goal="Find the cheapest red jacket and add to cart",
        task_id="demo.shopping.grounding",
        step=1,
        url="https://shopmart.example.com/search?q=red+jacket",
        max_entities=25,
        box_threshold=args.box_threshold,
        iou_threshold=args.iou_threshold,
        use_paddleocr=args.paddleocr,
        caption_icons=not args.no_caption,
    )
    return result


def run_head1(image: Image.Image) -> str:
    """Run Head 1 (heuristic) for comparison — needs the fake obs dict."""
    from browsergym_wrapper.example import (
        make_fake_axtree,
        make_fake_extra_props,
    )
    from browsergym_wrapper.heuristic import obs_to_schema

    obs = {
        "axtree_object": make_fake_axtree(),
        "extra_element_properties": make_fake_extra_props(),
        "focused_element_bid": "a2",
        "goal": "Find the cheapest red jacket and add to cart",
        "url": "https://shopmart.example.com/search?q=red+jacket",
        "last_action_error": "",
        "open_pages_urls": ["https://shopmart.example.com/search?q=red+jacket"],
    }
    return obs_to_schema(obs, step=1, task_id="demo.shopping.heuristic")


def main():
    parser = argparse.ArgumentParser(
        description="OmniParser-v2 grounding demo for BrowserGym (Head 3)",
    )
    parser.add_argument("--image", default=None, help="Path to screenshot PNG")
    parser.add_argument("--compare", action="store_true", help="Compare Head 1 vs Head 3")
    parser.add_argument("--save-annotated", default=None, help="Save annotated image to this path")
    parser.add_argument("--paddleocr", action="store_true", help="Use PaddleOCR")
    parser.add_argument("--no-caption", action="store_true", help="Skip Florence-2 icon captioning")
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    args = parser.parse_args()

    image = _load_or_generate_image(args.image)
    print(f"Image size: {image.size[0]}x{image.size[1]}")

    print("\n" + "=" * 60)
    print("HEAD 3: OmniParser-v2 Grounding (local, vision-only)")
    print("=" * 60)

    result = run_head3(image, args)
    elements = result["elements"]
    schema = result["schema"]

    print(f"\nDetected {len(elements)} UI elements:")
    for i, el in enumerate(elements):
        bbox = el.bbox
        print(f"  [{i:2d}] {el.element_type:5s} | {el.label:40s} | "
              f"({bbox.x},{bbox.y},{bbox.w},{bbox.h}) | "
              f"{'clickable' if el.interactable else 'static':>9s} | {el.source}")

    print(f"\nSchema ({len(schema)} chars):")
    print(textwrap.indent(schema, "  "))

    if result["warnings"]:
        print(f"\nWarnings: {result['warnings']}")

    if args.save_annotated:
        from vlm_wrapper.grounding import parse_screen_annotated
        _, annotated = parse_screen_annotated(
            image,
            box_threshold=args.box_threshold,
            iou_threshold=args.iou_threshold,
            use_paddleocr=args.paddleocr,
            caption_icons=not args.no_caption,
        )
        annotated.save(args.save_annotated)
        print(f"\nAnnotated image saved to: {args.save_annotated}")

    if args.compare:
        print("\n" + "=" * 60)
        print("HEAD 1: Heuristic (AXTree-based, free/instant)")
        print("=" * 60)

        try:
            h1_schema = run_head1(image)
            print(f"\nSchema ({len(h1_schema)} chars):")
            print(textwrap.indent(h1_schema, "  "))
        except Exception as e:
            print(f"\nHead 1 failed: {e}")

        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        from vlm_wrapper.schema import count_entities
        h3_count = count_entities(schema)
        try:
            h1_count = count_entities(h1_schema)
        except NameError:
            h1_count = 0
        print(f"  Head 1 (heuristic): {h1_count} entities  [AXTree → schema]")
        print(f"  Head 3 (grounding): {h3_count} entities  [pixels → OmniParser → schema]")
        print(f"  Head 3 raw detections: {len(elements)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
