#!/usr/bin/env python3
"""Unified GPT-4o VLM parser CLI.

Runs the ``vlm_wrapper`` GPT-4o (Head-2 / tool-loop) pipeline across
supported domains:

  * ``gymv``              — video game frames (2048, Sokoban, Minesweeper).
  * ``browser``           — BrowserGym / MiniWoB++ / WebArena screenshots.
  * ``desktop``           — OSWorld screenshots (via osworld_adapter).
  * ``tir_bench``         — TIR-Bench image QA (HF ``Agents-X/TIR-Bench``).
  * ``visual_toolbench``  — VisualToolBench (HF ``ScaleAI/VisualToolBench``).
  * ``video_holmes``      — Video-Holmes video-QA under ``data/Video-Holmes``.

The ``OPENAI_API_KEY`` is read from the environment (``.env`` is
auto-sourced when present). Override the model with ``--model`` or
``$VLM_LABEL_MODEL``.

Examples
--------

Parse a single game frame (no env required — a random frame is fine
for testing)::

    python scripts/run_vlm_parser.py gymv \
        --image vlm_wrapper/real_Games_Game2048-v0_step0.png \
        --goal "Reach 2048"

Parse one TIR-Bench test question::

    python scripts/run_vlm_parser.py tir_bench --limit 1

Parse 3 Video-Holmes test questions restricted to suspense-reasoning
(``SR``) type, saving results as JSONL::

    python scripts/run_vlm_parser.py video_holmes \
        --split test --limit 3 --question-types SR \
        --output out/vh_sr.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv(path: Path | None = None) -> None:
    """Minimal ``.env`` loader — no external dep.

    Only populates keys that are not already set so explicit environment
    variables always win. Ignores blank lines and ``#`` comments.
    """
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _write_output_json(path: str | None, payload: Any) -> None:
    if not path:
        return
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _print_schema_block(label: str, schema: str | None) -> None:
    print()
    print(f"-- {label} " + "-" * max(0, 70 - len(label)))
    if not schema:
        print("<no schema produced>")
        return
    # Strip any non-ASCII chars that the Windows default console (cp1252)
    # cannot render.  This is cosmetic only — the underlying schema is
    # still in ``out["schema"]`` / written verbatim to --output.
    try:
        print(schema.rstrip())
    except UnicodeEncodeError:
        print(schema.rstrip().encode("ascii", "replace").decode("ascii"))


# ======================================================================
# Sub-command handlers
# ======================================================================

def _parse_gymv(args: argparse.Namespace) -> int:
    from PIL import Image
    from gymv_wrapper.adapter import generate_label

    image = Image.open(args.image).convert("RGB")
    result = generate_label(
        image,
        goal=args.goal,
        task_id=args.task_id,
        step=args.step,
        game_rules=args.rules or "",
        obs_text=args.obs_text or "",
        max_entities=args.max_entities,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    _print_schema_block(f"gymv schema ({result['model']})", result["schema"])
    if result["warnings"]:
        print("warnings:", result["warnings"])
    _write_output_json(args.output, result)
    return 0 if result["schema"] else 1


def _parse_browser(args: argparse.Namespace) -> int:
    from PIL import Image
    from vlm_wrapper.browser_adapter import generate_label

    image = Image.open(args.image).convert("RGB")
    axtree_text = ""
    if args.axtree:
        axtree_text = Path(args.axtree).read_text(encoding="utf-8")

    result = generate_label(
        image,
        goal=args.goal,
        task_id=args.task_id,
        step=args.step,
        url=args.url or "",
        axtree_text=axtree_text,
        max_entities=args.max_entities,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    _print_schema_block(
        f"browser schema ({result['model']})", result["schema"],
    )
    if result["warnings"]:
        print("warnings:", result["warnings"])
    _write_output_json(args.output, result)
    return 0 if result["schema"] else 1


def _parse_desktop(args: argparse.Namespace) -> int:
    from PIL import Image
    from vlm_wrapper.osworld_adapter import generate_label

    image = Image.open(args.image).convert("RGB")
    a11y_xml = ""
    if args.a11y:
        a11y_xml = Path(args.a11y).read_text(encoding="utf-8")
    terminal_output = ""
    if args.terminal:
        terminal_output = Path(args.terminal).read_text(encoding="utf-8")

    result = generate_label(
        image,
        instruction=args.goal,
        goal=args.goal,
        task_id=args.task_id,
        step=args.step,
        a11y_tree_xml=a11y_xml,
        terminal_output=terminal_output,
        max_entities=args.max_entities,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    _print_schema_block(
        f"desktop schema ({result['model']})", result["schema"],
    )
    if result["warnings"]:
        print("warnings:", result["warnings"])
    _write_output_json(args.output, result)
    return 0 if result["schema"] else 1


def _parse_tir_bench(args: argparse.Namespace) -> int:
    from visual_reasoning_wrapper.benchmarks.tir_bench import (
        iter_tir_bench_samples, parse_tir_bench_batch, parse_tir_bench_sample,
    )

    samples = list(iter_tir_bench_samples(
        split=args.split,
        limit=args.limit,
    ))
    if not samples:
        print("No TIR-Bench samples matched the filter.")
        return 1

    if args.dry_run:
        for s in samples:
            print(f"id={s.sample_id} task={s.task}  Q: {s.prompt[:80]}…  A: {s.answer}")
        return 0

    if len(samples) == 1 and not args.output:
        out = parse_tir_bench_sample(
            samples[0],
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_entities=args.max_entities,
            max_rounds=args.max_rounds,
        )
        print(f"Question:     {samples[0].prompt}")
        print(f"Prediction:   {out['answer']}")
        print(f"Ground truth: {out['ground_truth']}")
        print(f"Correct:      {out['correct']}")
        _print_schema_block(f"TIR-Bench schema ({out['model']})", out["schema"])
        return 0 if out["schema"] else 1

    results = parse_tir_bench_batch(
        samples,
        output_jsonl=args.output,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_entities=args.max_entities,
        max_rounds=args.max_rounds,
    )
    correct = sum(1 for r in results if r.get("correct") is True)
    scored = sum(1 for r in results if r.get("correct") is not None)
    print(
        f"TIR-Bench: {correct}/{scored} correct "
        f"(out of {len(results)} attempted)"
    )
    return 0


def _parse_visual_toolbench(args: argparse.Namespace) -> int:
    from visual_reasoning_wrapper.benchmarks.visual_toolbench import (
        iter_visual_toolbench_samples,
        parse_visual_toolbench_batch,
        parse_visual_toolbench_sample,
    )

    samples = list(iter_visual_toolbench_samples(limit=args.limit))
    if not samples:
        print("No VisualToolBench samples matched the filter.")
        return 1

    if args.dry_run:
        for s in samples:
            print(f"id={s.sample_id}  Q: {s.question[:80]}…")
        return 0

    if len(samples) == 1 and not args.output:
        out = parse_visual_toolbench_sample(
            samples[0],
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_entities=args.max_entities,
            max_rounds=args.max_rounds,
        )
        print(f"Question:     {samples[0].question}")
        print(f"Prediction:   {out['answer']}")
        print(f"Ground truth: {out['ground_truth']}")
        print(f"Correct:      {out['correct']}")
        _print_schema_block(f"VisualToolBench schema ({out['model']})", out["schema"])
        return 0 if out["schema"] else 1

    results = parse_visual_toolbench_batch(
        samples,
        output_jsonl=args.output,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_entities=args.max_entities,
        max_rounds=args.max_rounds,
    )
    correct = sum(1 for r in results if r.get("correct") is True)
    scored = sum(1 for r in results if r.get("correct") is not None)
    print(
        f"VisualToolBench: {correct}/{scored} matched "
        f"(out of {len(results)} attempted; official eval uses rubrics)"
    )
    return 0


def _parse_video_holmes(args: argparse.Namespace) -> int:
    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples, parse_video_holmes_batch,
        parse_video_holmes_sample,
    )

    samples = list(iter_video_holmes_samples(
        split=args.split,
        video_holmes_root=args.video_holmes_root,
        limit=args.limit,
        question_types=args.question_types or None,
    ))
    if not samples:
        print("No Video-Holmes samples matched the filter.")
        return 1

    if args.dry_run:
        for s in samples:
            video_status = "present" if s.video_path else "missing"
            print(
                f"{s.video_id}  Q{s.question_id}  type={s.question_type}  "
                f"video={video_status}  "
                f"Q: {s.question[:60]}  gt={s.answer}"
            )
        return 0

    if len(samples) == 1 and not args.output:
        out = parse_video_holmes_sample(
            samples[0],
            num_frames=args.num_frames,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_entities=args.max_entities,
            max_rounds=args.max_rounds,
        )
        s = samples[0]
        print(f"Video:        {s.video_id}  Q{s.question_id}  ({s.question_type})")
        print(f"Question:     {s.question}")
        print(f"Prediction:   {out['answer']}  (raw: {out['answer_raw']!r})")
        print(f"Ground truth: {out['ground_truth']}")
        print(f"Correct:      {out['correct']}")
        print(f"Frames sent:  {out['num_frames']}")
        _print_schema_block(
            f"Video-Holmes schema ({out['model']})", out["schema"],
        )
        return 0 if out["schema"] else 1

    results = parse_video_holmes_batch(
        samples,
        output_jsonl=args.output,
        num_frames=args.num_frames,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_entities=args.max_entities,
        max_rounds=args.max_rounds,
    )
    correct = sum(1 for r in results if r.get("correct") is True)
    scored = sum(1 for r in results if r.get("correct") is not None)
    print(
        f"Video-Holmes {args.split}: {correct}/{scored} correct "
        f"(out of {len(results)} attempted)"
    )
    return 0


# ======================================================================
# Argument parsing
# ======================================================================

def _add_common_vlm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None,
                   help="Vision model (default: $VLM_LABEL_MODEL or gpt-4o)")
    p.add_argument("--api-key", default=None,
                   help="OpenAI API key (default: $OPENAI_API_KEY)")
    p.add_argument("--base-url", default=None,
                   help="OpenAI base URL override (OpenRouter, Azure, ...)")
    p.add_argument("--max-entities", type=int, default=20)
    p.add_argument("--max-rounds", type=int, default=4,
                   help="Max VLM tool-calling rounds (image/video QA)")
    p.add_argument("--output", default=None,
                   help="Write the structured result to this path "
                        "(JSON for single-sample runs, JSONL for batches).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the vlm_wrapper GPT-4o parser across "
                    "gymv / browser / desktop / tir_bench / visual_toolbench / video_holmes.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    gymv = sub.add_parser("gymv", help="Parse a Gym-V game frame.")
    gymv.add_argument("--image", required=True, help="Path to the game frame PNG.")
    gymv.add_argument("--goal", default="", help="Short goal description.")
    gymv.add_argument("--task-id", default="",
                      help="e.g. Game2048-v0")
    gymv.add_argument("--step", type=int, default=0)
    gymv.add_argument("--rules", default="",
                      help="Optional: full env.description text (game rules).")
    gymv.add_argument("--obs-text", default="",
                      help="Optional: native obs.text for grounding context.")
    _add_common_vlm_args(gymv)
    gymv.set_defaults(func=_parse_gymv)

    browser = sub.add_parser("browser", help="Parse a BrowserGym screenshot.")
    browser.add_argument("--image", required=True,
                         help="Path to the browser screenshot PNG.")
    browser.add_argument("--goal", default="", help="Task instruction.")
    browser.add_argument("--url", default="", help="Current page URL.")
    browser.add_argument("--axtree", default=None,
                         help="Path to a text file containing the AXTree dump.")
    browser.add_argument("--task-id", default="")
    browser.add_argument("--step", type=int, default=0)
    _add_common_vlm_args(browser)
    browser.set_defaults(func=_parse_browser)

    desktop = sub.add_parser("desktop", help="Parse an OSWorld screenshot.")
    desktop.add_argument("--image", required=True,
                         help="Path to the desktop screenshot PNG.")
    desktop.add_argument("--goal", default="", help="OSWorld task instruction.")
    desktop.add_argument("--a11y", default=None,
                         help="Path to accessibility-tree XML (optional).")
    desktop.add_argument("--terminal", default=None,
                         help="Path to terminal-output text (optional).")
    desktop.add_argument("--task-id", default="")
    desktop.add_argument("--step", type=int, default=0)
    _add_common_vlm_args(desktop)
    desktop.set_defaults(func=_parse_desktop)

    tir = sub.add_parser("tir_bench", help="Parse TIR-Bench image-QA (HF).")
    tir.add_argument("--split", default="test", choices=["test"],
                     help="TIR-Bench only ships the test split.")
    tir.add_argument("--limit", type=int, default=3)
    tir.add_argument("--dry-run", action="store_true",
                     help="List samples without calling the VLM.")
    _add_common_vlm_args(tir)
    tir.set_defaults(func=_parse_tir_bench)

    vtb = sub.add_parser("visual_toolbench", help="Parse VisualToolBench (HF).")
    vtb.add_argument("--limit", type=int, default=3)
    vtb.add_argument("--dry-run", action="store_true",
                     help="List samples without calling the VLM.")
    _add_common_vlm_args(vtb)
    vtb.set_defaults(func=_parse_visual_toolbench)

    vh = sub.add_parser("video_holmes", help="Parse Video-Holmes samples.")
    vh.add_argument("--split", default="test", choices=["test", "train"])
    vh.add_argument("--limit", type=int, default=3)
    vh.add_argument("--num-frames", type=int, default=8)
    vh.add_argument("--question-types", nargs="+", default=None,
                    help="Filter to these question types (SR IMC TCI TA MHR "
                         "PAR CTI).")
    vh.add_argument("--video-holmes-root", default=None,
                    help="Override data/Video-Holmes path.")
    vh.add_argument("--dry-run", action="store_true",
                    help="List samples without calling GPT-4o.")
    _add_common_vlm_args(vh)
    vh.set_defaults(func=_parse_video_holmes)

    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()

    logging.basicConfig(
        level=os.environ.get("VLM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "api_key", None):
        args.api_key = os.environ.get("OPENAI_API_KEY")
    if not getattr(args, "model", None):
        args.model = os.environ.get("VLM_LABEL_MODEL", "gpt-4o")

    # dry-run modes never need a key
    dry_run = getattr(args, "dry_run", False)
    if not dry_run and not args.api_key:
        print(
            "No OPENAI_API_KEY found. Put one in .env or set the environment "
            "variable before running the VLM parser.",
            file=sys.stderr,
        )
        return 2

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
