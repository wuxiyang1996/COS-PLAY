"""Smoke test for the unified `vlm_benchmarks` conda env.

Checks each of the 5 domain stacks listed in PLAN-VISUAL-GROUNDING.md
can be imported and used for basic operations.
"""
from __future__ import annotations

import sys
import traceback

failures: list[tuple[str, str]] = []
warnings: list[tuple[str, str]] = []


def check(label: str, fn, required: bool = True) -> None:
    try:
        out = fn()
        msg = f"           {out}" if out else ""
        print(f"  [OK]   {label}{(': ' + out) if out else ''}")
    except Exception as exc:
        tb = traceback.format_exc(limit=1).splitlines()[-1]
        if required:
            failures.append((label, str(exc)))
            print(f"  [FAIL] {label}: {exc}")
        else:
            warnings.append((label, str(exc)))
            print(f"  [WARN] {label}: {exc}  (optional)")


print(f"Python {sys.version.split()[0]}")
print(f"executable: {sys.executable}\n")

# ------- Core ML stack (shared across all domains) -------
print("Core ML:")
check("torch",                    lambda: __import__('torch').__version__)
check("torch.cuda",               lambda: f"available={__import__('torch').cuda.is_available()} devices={__import__('torch').cuda.device_count()}")
check("torchvision",              lambda: __import__('torchvision').__version__)
check("transformers",             lambda: __import__('transformers').__version__)
check("accelerate",               lambda: __import__('accelerate').__version__)
check("huggingface_hub",          lambda: __import__('huggingface_hub').__version__)
check("datasets",                 lambda: __import__('datasets').__version__)
check("sentence_transformers",    lambda: __import__('sentence_transformers').__version__)
print()

# ------- vlm_wrapper vision backends -------
print("vlm_wrapper vision backends:")
check("timm (Florence-2 backbone)",   lambda: __import__('timm').__version__)
check("easyocr",                      lambda: __import__('easyocr').__version__)
check("ultralytics (YOLO)",           lambda: __import__('ultralytics').__version__)
check("opencv (cv2)",                 lambda: __import__('cv2').__version__)
check("decord",                       lambda: __import__('decord').__version__)
check("av",                           lambda: __import__('av').__version__)
check("supervision",                  lambda: __import__('supervision').__version__)
check("PIL",                          lambda: __import__('PIL').__version__)
print()

# ------- Domain 1: gymv -------
print("Domain 1 — gymv:")
check("gym",                          lambda: __import__('gym').__version__)
check("gymnasium",                    lambda: __import__('gymnasium').__version__)
check("gamingagent",                  lambda: __import__('gamingagent').__version__ if hasattr(__import__('gamingagent'), '__version__') else "(imported)", required=False)
print()

# ------- Domain 2: browser -------
print("Domain 2 — browser:")
check("browsergym",                   lambda: (__import__('browsergym', fromlist=['*']), "imported")[-1])
check("browsergym.core",              lambda: (__import__('browsergym.core', fromlist=['*']), "imported")[-1])
check("browsergym.miniwob",           lambda: (__import__('browsergym.miniwob', fromlist=['*']), "imported")[-1])
check("browsergym.webarena",          lambda: (__import__('browsergym.webarena', fromlist=['*']), "imported")[-1])
check("browsergym.visualwebarena",    lambda: (__import__('browsergym.visualwebarena', fromlist=['*']), "imported")[-1])
check("browsergym.workarena",         lambda: (__import__('browsergym.workarena', fromlist=['*']), "imported")[-1], required=False)
check("playwright",                   lambda: __import__('playwright').__version__ if hasattr(__import__('playwright'), '__version__') else "(imported)")
def _chromium_probe() -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        b.close()
    return "chromium launches headless"
check("playwright chromium",          _chromium_probe)
print()

# ------- Domain 3: desktop (grounding only; desktop-env lives in `osworld`) -------
print("Domain 3 — desktop (grounding adapter):")
check("docker SDK",                   lambda: __import__('docker').__version__)
def _desktop_adapter() -> str:
    from vlm_wrapper.grounding_browsergym import grounding_osworld_obs_to_schema  # type: ignore
    return "grounding_osworld_obs_to_schema importable"
check("OSWorld grounding adapter",    _desktop_adapter)
print()

# ------- Domain 4: image_qa -------
print("Domain 4 — image_qa:")
def _tir_bench_loader() -> str:
    from visual_reasoning_wrapper.benchmarks import tir_bench as _tb  # type: ignore  # noqa: F401
    return "visual_reasoning_wrapper.benchmarks.tir_bench importable"
check("TIR-Bench loader",             _tir_bench_loader)
check("openai client",                lambda: __import__('openai').__version__)
check("anthropic client",             lambda: __import__('anthropic').__version__)
check("google.genai client",          lambda: (__import__('google.genai', fromlist=['*']), "imported")[-1])
print()

# ------- Domain 5: video_qa -------
print("Domain 5 — video_qa:")
def _videoholmes_loader() -> str:
    from visual_reasoning_wrapper.benchmarks import video_holmes as _vh  # type: ignore  # noqa: F401
    return "visual_reasoning_wrapper.benchmarks.video_holmes importable"
check("Video-Holmes loader",          _videoholmes_loader)
def _decord_readable() -> str:
    import decord, numpy as np
    return f"decord {decord.__version__} loaded"
check("decord frame reader",          _decord_readable)
print()

# ------- vlm_wrapper grounding pipeline entry points -------
print("vlm_wrapper grounding pipeline:")
def _ground_import() -> str:
    from vlm_wrapper import GroundingRequest, cascaded_ground  # type: ignore
    return "GroundingRequest + cascaded_ground importable"
check("vlm_wrapper.ground",           _ground_import)
def _schema_import() -> str:
    from vlm_wrapper.schema import build_adaptive_system_prompt, validate_schema  # type: ignore
    return "schema module importable"
check("vlm_wrapper.schema",           _schema_import)
print()

# ------- Data reachability -------
print("Benchmark data on disk:")
from pathlib import Path
ROOT = Path("/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/data")

def _tir_bench_hf_data() -> str:
    from datasets import load_dataset

    ds = load_dataset("Agents-X/TIR-Bench", split="test", trust_remote_code=True)
    return f"TIR-Bench HF test rows={len(ds)}"
check("TIR-Bench HF data",            _tir_bench_hf_data, required=False)

def _videoholmes_data() -> str:
    bench = ROOT / "Video-Holmes" / "Benchmark"
    questions = bench / "test_Video-Holmes.json"
    videos = bench / "videos" / "videos_cropped"
    if not questions.exists():
        raise FileNotFoundError(f"{questions}")
    n_vids = len(list(videos.glob("*.mp4"))) if videos.is_dir() else 0
    return f"{n_vids} videos, test_Video-Holmes.json present"
check("Video-Holmes data",            _videoholmes_data, required=False)
print()

# ------- End-to-end grounding: image_qa synthetic -------
print("End-to-end grounding pipeline (offline, no API call):")
def _ground_synthetic() -> str:
    from PIL import Image, ImageDraw
    from vlm_wrapper import GroundingRequest
    from vlm_wrapper.schema import build_adaptive_system_prompt
    img = Image.new("RGB", (256, 256), color="white")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 120, 120], fill="red")
    req = GroundingRequest(images=img, goal="Count red squares.", domain="image_qa")
    prompt = build_adaptive_system_prompt(
        domain="image_qa",
        sections=["entities", "attributes", "state_flags", "targets", "evidence", "answer"],
        task_type="qa",
        max_entities=8,
    )
    return f"GroundingRequest OK, adaptive prompt len={len(prompt)}"
check("image_qa prompt build",        _ground_synthetic)
print()

# ------- Summary -------
print("=" * 60)
if failures:
    print(f"{len(failures)} REQUIRED check(s) FAILED:")
    for label, err in failures:
        print(f"  ✗ {label}: {err}")
    sys.exit(1)
print("All required checks passed.")
if warnings:
    print(f"{len(warnings)} optional check(s) flagged:")
    for label, err in warnings:
        print(f"  ⚠ {label}: {err}")
print("=" * 60)
