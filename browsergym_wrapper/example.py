#!/usr/bin/env python3
"""Example: browsergym_wrapper on a simulated BrowserGym observation.

Demonstrates both heads of the BrowserGym pipeline:

  Head 1 (Heuristic): AXTree dict → structured schema  [FREE, instant]
  Head 2 (Vision):    Screenshot  → GPT-4o → schema     [requires API key]

No BrowserGym installation required — we build a realistic obs dict by
hand, mimicking what BrowserEnv._get_obs() returns for a shopping page.

Usage::

    # Head 1 only (no API key needed)
    python -m browsergym_wrapper.example

    # Both heads (needs OPENAI_API_KEY or --api-key)
    python -m browsergym_wrapper.example --vision --api-key sk-...

    # Use OpenRouter
    python -m browsergym_wrapper.example --vision \\
        --api-key $OPENROUTER_KEY \\
        --base-url https://openrouter.ai/api/v1 \\
        --model openai/gpt-4.1
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browsergym_wrapper.heuristic import obs_to_schema
from vlm_wrapper.schema import validate_schema


# =====================================================================
# Build a fake BrowserGym observation dict
# =====================================================================

def make_fake_axtree() -> dict:
    """Construct a minimal AXTree dict resembling BrowserGym's axtree_object.

    This mimics a product search results page with a search bar, nav
    links, product cards, and an "Add to Cart" button.
    """
    nodes = [
        {"nodeId": "1", "role": {"value": "WebArea"}, "name": {"value": "ShopMart"},
         "childIds": ["2", "3", "10"], "properties": []},
        {"nodeId": "2", "role": {"value": "navigation"}, "name": {"value": "Main Menu"},
         "browsergym_id": "a1", "childIds": ["4", "5"], "properties": []},
        {"nodeId": "3", "role": {"value": "searchbox"}, "name": {"value": "Search products"},
         "browsergym_id": "a2", "childIds": [], "properties": [],
         "value": {"value": "red jacket"}},
        {"nodeId": "4", "role": {"value": "link"}, "name": {"value": "Men's Clothing"},
         "browsergym_id": "a3", "childIds": [], "properties": []},
        {"nodeId": "5", "role": {"value": "link"}, "name": {"value": "Women's Clothing"},
         "browsergym_id": "a4", "childIds": [], "properties": []},
        {"nodeId": "10", "role": {"value": "main"}, "name": {"value": "Search Results"},
         "browsergym_id": "b1", "childIds": ["11", "12", "13", "14"], "properties": []},
        {"nodeId": "11", "role": {"value": "combobox"}, "name": {"value": "Sort By"},
         "browsergym_id": "b2", "childIds": [], "properties": [],
         "value": {"value": "Relevance"}},
        {"nodeId": "12", "role": {"value": "link"},
         "name": {"value": "Red Wool Jacket - $89.99"},
         "browsergym_id": "c1", "childIds": [], "properties": []},
        {"nodeId": "13", "role": {"value": "link"},
         "name": {"value": "Red Down Jacket - $49.99"},
         "browsergym_id": "c2", "childIds": [], "properties": []},
        {"nodeId": "14", "role": {"value": "button"},
         "name": {"value": "Add to Cart"},
         "browsergym_id": "c3", "childIds": [], "properties": []},
    ]
    return {"nodes": nodes}


def make_fake_extra_props() -> dict:
    """Bounding boxes and interaction flags for each element bid."""
    return {
        "a1": {"visibility": 1.0, "bbox": [0, 0, 1200, 50], "clickable": False, "set_of_marks": False},
        "a2": {"visibility": 1.0, "bbox": [400, 8, 300, 34], "clickable": True, "set_of_marks": True},
        "a3": {"visibility": 1.0, "bbox": [20, 12, 120, 26], "clickable": True, "set_of_marks": True},
        "a4": {"visibility": 1.0, "bbox": [150, 12, 140, 26], "clickable": True, "set_of_marks": True},
        "b1": {"visibility": 1.0, "bbox": [0, 60, 1200, 700], "clickable": False, "set_of_marks": False},
        "b2": {"visibility": 1.0, "bbox": [900, 70, 150, 30], "clickable": True, "set_of_marks": True},
        "c1": {"visibility": 1.0, "bbox": [50, 130, 350, 280], "clickable": True, "set_of_marks": True},
        "c2": {"visibility": 1.0, "bbox": [450, 130, 350, 280], "clickable": True, "set_of_marks": True},
        "c3": {"visibility": 1.0, "bbox": [500, 420, 140, 40], "clickable": True, "set_of_marks": True},
    }


def make_fake_screenshot(w: int = 1200, h: int = 800) -> np.ndarray:
    """Draw a crude but recognizable shopping page screenshot."""
    img = Image.new("RGB", (w, h), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_sm = ImageFont.truetype("arial.ttf", 13)
        font_lg = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font
        font_lg = font

    draw.rectangle([0, 0, w, 50], fill="#2C3E50")
    draw.text((20, 14), "Men's Clothing", fill="#ECF0F1", font=font_sm)
    draw.text((160, 14), "Women's Clothing", fill="#ECF0F1", font=font_sm)
    draw.rectangle([400, 8, 700, 42], fill="#FFFFFF", outline="#BDC3C7")
    draw.text((410, 14), "red jacket", fill="#2C3E50", font=font_sm)
    draw.text((1050, 14), "ShopMart", fill="#E74C3C", font=font_lg)

    draw.text((50, 70), 'Search Results for "red jacket"', fill="#2C3E50", font=font_lg)

    draw.rectangle([900, 70, 1050, 100], outline="#BDC3C7")
    draw.text((910, 76), "Sort: Relevance", fill="#7F8C8D", font=font_sm)

    draw.rectangle([50, 130, 400, 410], outline="#BDC3C7", width=2)
    draw.rectangle([60, 140, 390, 310], fill="#E74C3C")
    draw.text((170, 210), "JACKET", fill="#FFFFFF", font=font_lg)
    draw.text((60, 320), "Red Wool Jacket", fill="#2C3E50", font=font)
    draw.text((60, 345), "$89.99", fill="#E74C3C", font=font_lg)
    draw.rectangle([60, 375, 200, 400], fill="#3498DB")
    draw.text((80, 379), "Add to Cart", fill="#FFFFFF", font=font_sm)

    draw.rectangle([450, 130, 800, 410], outline="#BDC3C7", width=2)
    draw.rectangle([460, 140, 790, 310], fill="#C0392B")
    draw.text((570, 210), "JACKET", fill="#FFFFFF", font=font_lg)
    draw.text((460, 320), "Red Down Jacket", fill="#2C3E50", font=font)
    draw.text((460, 345), "$49.99", fill="#27AE60", font=font_lg)
    draw.rectangle([460, 375, 600, 400], fill="#27AE60")
    draw.text((480, 379), "Add to Cart", fill="#FFFFFF", font=font_sm)

    draw.rectangle([700, 140, 790, 165], fill="#F39C12")
    draw.text((710, 144), "CHEAPEST", fill="#FFFFFF", font=font_sm)

    return np.array(img)


def build_browsergym_obs(
    goal: str = "Find the cheapest red jacket and add it to cart",
) -> dict:
    """Assemble a dict that looks like BrowserEnv._get_obs() output."""
    return {
        "axtree_object": make_fake_axtree(),
        "dom_object": None,
        "extra_element_properties": make_fake_extra_props(),
        "screenshot": make_fake_screenshot(),
        "focused_element_bid": "a2",
        "url": "https://shopmart.example.com/search?q=red+jacket",
        "goal": goal,
        "goal_object": [],
        "open_pages_urls": ("https://shopmart.example.com/search?q=red+jacket",),
        "last_action": 'fill(a2, "red jacket")',
        "last_action_error": "",
        "elapsed_time": 4.2,
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="browsergym_wrapper example (no BrowserGym install needed)")
    parser.add_argument("--vision", action="store_true",
                        help="Also run Head 2 (vision -> GPT-4o). Needs API key.")
    parser.add_argument("--api-key", default=None, help="OpenAI / OpenRouter API key")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--model", default=None,
                        help="Vision model (default: gpt-4o or $VLM_LABEL_MODEL)")
    parser.add_argument("--save-image", action="store_true",
                        help="Save the synthetic screenshot to disk")
    args = parser.parse_args()

    obs = build_browsergym_obs()
    sep = "=" * 70

    if args.save_image:
        img = Image.fromarray(obs["screenshot"])
        out_path = Path(__file__).parent / "example_screenshot.png"
        img.save(str(out_path))
        print(f"Saved screenshot -> {out_path}")

    print(f"\n{sep}")
    print("  HEAD 1 - Heuristic (AXTree -> schema)")
    print("  Cost: $0 | Latency: <1 ms")
    print(sep)

    schema = obs_to_schema(obs, step=3, task_id="shopmart.search.jacket")
    print(schema)

    warnings = validate_schema(schema)
    if warnings:
        print(f"\n[!] Validation warnings: {warnings}")
    else:
        print("\nValidation: PASSED")

    if args.vision:
        from browsergym_wrapper.adapter import generate_label
        import time

        print(f"\n{sep}")
        print("  HEAD 2 - Vision (screenshot -> GPT-4o -> schema)")
        model_name = args.model or "gpt-4o"
        print(f"  Model: {model_name}")
        print(sep)

        img = Image.fromarray(obs["screenshot"])
        t0 = time.time()
        result = generate_label(
            img,
            goal=obs["goal"],
            task_id="shopmart.search.jacket",
            step=3,
            url=obs["url"],
            last_action=obs["last_action"],
            last_action_error=obs["last_action_error"],
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        elapsed = round(time.time() - t0, 2)

        if result["schema"]:
            print(result["schema"])
        else:
            print("[NO SCHEMA PARSED]")
            print("Raw output:")
            print(textwrap.indent(result["raw"][:2000], "  "))

        if result["warnings"]:
            print(f"\nWarnings: {result['warnings']}")
        else:
            print("\nValidation: PASSED")
        print(f"Elapsed: {elapsed}s")

    print(f"\n{sep}")
    print("  What happened:")
    print("  1. Built a fake BrowserGym obs dict (AXTree + bboxes + screenshot)")
    print("  2. Ran Head 1 (heuristic): parsed AXTree -> <state>...</state>")
    if args.vision:
        print("  3. Ran Head 2 (vision): sent screenshot -> GPT -> <state>...</state>")
    else:
        print("  Tip: add --vision to also test the GPT-4o vision head")
    print(sep)


if __name__ == "__main__":
    main()
