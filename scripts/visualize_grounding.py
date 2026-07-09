"""End-to-end visual grounding runner with rich per-case artifacts.

Runs the tool-calling grounding pipeline over all five domains (gymv,
browser, desktop, image_qa/TIR-Bench, video_qa/Video-Holmes) with GPT-4o and
writes a self-contained artifact bundle per run:

  out/grounding_runs/<timestamp>/
    index.html                 – gallery linking to every case
    summary.json               – aggregate scorecard
    <case>/
      input.png                – the screenshot / frame sent to GPT-4o
      annotated.png            – input + bbox/label overlay for each
                                 <entity> that has a pos=
      schema.txt               – the <state>...</state> block
      tool_trace.json          – [{call:{name,arguments}, result}, …]
      case.json                – the full per-case record
      trace_summary.md         – human-readable run report

The driver delegates per-case grounding to the existing
`scripts/test_vlm_parsers.py` runners so the behaviour matches the CI
scorecard exactly — this script only adds visualisation.

Usage::

    python scripts/visualize_grounding.py \\
        --cases gymv browser desktop tir_bench video_holmes \\
        --gymv-head tool_loop --browser-head tool_loop --desktop-head tool_loop \\
        --model gpt-4o --max-rounds 8

Environment:
    OPENAI_API_KEY (required)   OPENAI_BASE_URL (optional)
    HF_HOME (optional, for OmniParser/Florence-2 weight cache)

Outputs are human-inspectable — open ``index.html`` in any browser.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import test_vlm_parsers as tvp  # type: ignore

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("visualize_grounding")


# ── Entity / bbox extraction ─────────────────────────────────────────

_ENTITY_LINE_RE = re.compile(
    r"^(e\d+)\s*\[(.*?)\]\s*$",
    re.MULTILINE,
)
_POS_FIELD_RE = re.compile(
    r"pos\s*=\s*(null|\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)",
)
_LABEL_FIELD_RE = re.compile(r'label\s*=\s*"?([^,"]+?)"?(?:\s*,|\s*$)')
_TYPE_FIELD_RE = re.compile(r"(?:^|[,\s\[])type\s*=\s*([a-zA-Z_]+)")


def _parse_entities(schema: str) -> list[dict[str, Any]]:
    """Return a list of ``{"id","label","type","pos":(x,y,w,h)|None}``."""
    if not schema:
        return []
    ents: list[dict[str, Any]] = []
    # Only look inside <entities> — tile identifiers are reused in
    # <attributes>/<uncertainty> without bbox info and we don't want to
    # re-surface those as duplicate overlay boxes.
    m = re.search(r"<entities>(.*?)(?:<[a-z_]+>|$)", schema, re.DOTALL)
    body = m.group(1) if m else schema

    for line_m in _ENTITY_LINE_RE.finditer(body):
        eid = line_m.group(1)
        inline = line_m.group(2)
        pos_m = _POS_FIELD_RE.search(inline)
        pos: tuple[int, int, int, int] | None = None
        if pos_m and pos_m.group(1) != "null":
            nums = [int(n.strip()) for n in pos_m.group(1).split(",")]
            if len(nums) == 4:
                pos = (nums[0], nums[1], nums[2], nums[3])
        label_m = _LABEL_FIELD_RE.search(inline)
        label = label_m.group(1).strip() if label_m else ""
        type_m = _TYPE_FIELD_RE.search(inline)
        etype = type_m.group(1).strip() if type_m else "?"
        ents.append({"id": eid, "label": label, "type": etype, "pos": pos})
    return ents


# ── Annotation rendering ─────────────────────────────────────────────

_TYPE_COLOURS = {
    "element": (30, 144, 255),   # dodger blue
    "object":  (220, 60, 60),    # red
    "region":  (60, 200, 60),    # green
    "text":    (255, 165, 0),    # orange
    "icon":    (170, 90, 220),   # purple (aliased)
    "?":       (128, 128, 128),
}


def _annotate(
    image: Image.Image,
    entities: list[dict[str, Any]],
    case: str,
) -> Image.Image:
    """Draw bbox + id + label overlays onto a copy of *image*."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 14)
        small = ImageFont.truetype(
            "/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    # gymv pos is grid coordinates (r, c, 1, 1); scale them onto pixels.
    is_gymv = case == "gymv"
    W, H = img.size
    grid_cell_w, grid_cell_h = W / 4, H / 4  # 2048 is a 4×4 grid

    painted = 0
    for ent in entities:
        pos = ent["pos"]
        if pos is None:
            continue
        if is_gymv:
            r, c, _, _ = pos
            x, y = int(c * grid_cell_w) + 4, int(r * grid_cell_h) + 4
            w, h = int(grid_cell_w) - 8, int(grid_cell_h) - 8
        else:
            x, y, w, h = pos
        if w <= 0 or h <= 0:
            continue
        x2, y2 = x + w, y + h
        colour = _TYPE_COLOURS.get(ent["type"].lower(), _TYPE_COLOURS["?"])

        for dx in range(3):
            draw.rectangle(
                [x - dx, y - dx, x2 + dx, y2 + dx],
                outline=colour,
            )
        tag = f"{ent['id']} {ent['label']}"[:40]
        tw = max(8, int(len(tag) * 7))
        draw.rectangle([x, y - 18, x + tw, y], fill=colour + (230,))
        draw.text((x + 2, y - 17), tag, fill="white", font=small)
        painted += 1

    title = f"{case}  |  {painted}/{len(entities)} entities drawn"
    tw = max(8, int(len(title) * 7))
    draw.rectangle([0, 0, tw + 6, 20], fill=(0, 0, 0, 200))
    draw.text((4, 3), title, fill="white", font=font)
    return img


# ── Per-case driver (wraps test_vlm_parsers._run_case) ─────────────

def _case_input_path(case: str, capture_dir: Path) -> Path | None:
    """Return the PNG the corresponding `_capture_or_synthesize_*`
    helper wrote, so we can copy it into the run bundle."""
    mapping = {
        "gymv":         "gymv_2048.png",
        "browser":      "browser_wiki.png",
        "desktop":      "desktop.png",
    }
    fname = mapping.get(case)
    if fname and (capture_dir / fname).exists():
        return capture_dir / fname
    return None


def _copy_tir_bench_input(case_result: dict[str, Any], dst: Path) -> bool:
    """``test_vlm_parsers`` saves the decoded frame to ``out/captures``."""
    src_s = case_result.get("input_image_path")
    if not src_s:
        return False
    src = Path(src_s)
    if src.is_file():
        shutil.copy(src, dst)
        return True
    return False


def _copy_video_first_frame(case_result: dict[str, Any], dst: Path) -> bool:
    """Video-Holmes bundles only keep `video_id`; re-resolve the video
    file via the benchmark helpers and render its first frame as the
    representative screenshot."""
    video_id = case_result.get("video_id")
    if not video_id:
        return False
    try:
        from visual_reasoning_wrapper.benchmarks.video_holmes import (
            _video_path, default_video_holmes_root,
        )
        video = _video_path(default_video_holmes_root(), video_id)
    except Exception:
        return False
    if not video or not Path(video).exists():
        return False
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(str(video))
        frame = vr[0].asnumpy()
        Image.fromarray(frame).save(dst)
        return True
    except Exception as exc:
        logger.debug("decord fallback failed for %s: %s", video, exc)
        try:
            import av  # type: ignore
            container = av.open(str(video))
            for frame in container.decode(video=0):
                frame.to_image().save(dst)
                break
            container.close()
            return True
        except Exception as exc2:
            logger.warning("video frame extraction failed: %s", exc2)
            return False


def _render_case_md(
    case: str,
    case_result: dict[str, Any],
    schema: str | None,
    has_annot: bool,
) -> str:
    val = case_result.get("validation") or {}
    head = case_result.get("head_used", "?")
    entc = val.get("entity_count", 0)
    valid = val.get("valid")
    trace = case_result.get("tool_trace") or []
    lines = [
        f"# {case}",
        "",
        f"- **head_used:** `{head}`",
        f"- **rounds:** {case_result.get('rounds', '?')}",
        f"- **entities:** {entc}",
        f"- **schema valid:** {valid}",
        f"- **elapsed:** {case_result.get('elapsed_s', '?')} s",
    ]
    if "correct" in case_result:
        lines.extend([
            f"- **prediction:** `{case_result.get('answer')!r}`",
            f"- **ground_truth:** `{case_result.get('ground_truth')!r}`",
            f"- **correct:** {case_result.get('correct')}",
        ])
    lines.extend([
        "",
        "## Tool calls",
        "",
    ])
    if not trace:
        lines.append("_(no tool calls — single-shot VLM head)_")
    else:
        for i, tc in enumerate(trace, 1):
            call = tc.get("call") or {}
            args = call.get("arguments") or {}
            fn = call.get("name", "?")
            res = tc.get("result")
            res_preview = json.dumps(res, default=str)[:250]
            lines.append(
                f"{i}. `{fn}({json.dumps(args, default=str)[:120]})`  "
                f"→ `{res_preview}{'…' if len(res_preview) == 250 else ''}`"
            )
    lines.extend([
        "",
        "## Schema",
        "",
        "```",
        schema or "<no schema produced>",
        "```",
        "",
    ])
    if val.get("errors"):
        lines.append("## Validation errors")
        for e in val["errors"]:
            lines.append(f"- {e}")
        lines.append("")
    if val.get("warnings"):
        lines.append("## Validation warnings")
        for w in val["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    if has_annot:
        lines.extend([
            "## Artifacts",
            "- `input.png` — raw screenshot / frame",
            "- `annotated.png` — input with `<entities>` bboxes overlaid",
            "- `schema.txt` — the `<state>` block",
            "- `tool_trace.json` — raw tool call trace",
            "- `case.json` — full per-case record",
        ])
    return "\n".join(lines) + "\n"


def _render_index_html(
    run_dir: Path,
    summaries: list[dict[str, Any]],
) -> None:
    cards = []
    for s in summaries:
        case = s.get("case")
        if not case:
            continue
        case_dir = run_dir / case
        val = s.get("validation") or {}
        head = s.get("head_used", "?")
        ents = val.get("entity_count", 0)
        rounds = s.get("rounds", "?")
        trace_len = len(s.get("tool_trace") or [])
        valid = val.get("valid")
        correct = s.get("correct")
        verdict = "—" if correct is None else ("CORRECT" if correct else "WRONG")
        valid_badge = "OK" if valid else "FAIL"
        extra = ""
        if correct is not None:
            extra = (
                f"<div class='meta'>prediction: "
                f"<code>{s.get('answer')!r}</code> · "
                f"gt: <code>{s.get('ground_truth')!r}</code> "
                f"→ <b>{verdict}</b></div>"
            )

        annot = case_dir / "annotated.png"
        input_img = case_dir / "input.png"
        img_src = (
            f"{case}/annotated.png"
            if annot.exists()
            else (f"{case}/input.png" if input_img.exists() else "")
        )
        img_html = (
            f'<a href="{img_src}"><img src="{img_src}" alt="{case}"></a>'
            if img_src else "<div class='noimg'>(no image)</div>"
        )

        cards.append(f"""
<section class='card'>
  <header>
    <h2>{case}</h2>
    <span class='pill pill-{"ok" if valid else "fail"}'>{valid_badge}</span>
  </header>
  {img_html}
  <div class='meta'>
    head=<code>{head}</code> · rounds={rounds} · tool_calls={trace_len} ·
    entities={ents} · elapsed={s.get('elapsed_s','?')}s
  </div>
  {extra}
  <div class='links'>
    <a href='{case}/schema.txt'>schema.txt</a> ·
    <a href='{case}/tool_trace.json'>tool_trace.json</a> ·
    <a href='{case}/trace_summary.md'>trace_summary.md</a> ·
    <a href='{case}/case.json'>case.json</a>
  </div>
</section>""")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'>
<title>vlm_wrapper grounding run — {ts}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        max-width: 1200px; margin: 20px auto; padding: 0 16px;
        background: #fafafa; color: #111; }}
h1 {{ margin: 0 0 8px; }}
header.page {{ display: flex; align-items: baseline; gap: 16px;
               border-bottom: 1px solid #ddd; padding-bottom: 12px;
               margin-bottom: 16px; }}
header.page .sub {{ color: #555; font-size: 0.9em; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill,
         minmax(420px, 1fr)); gap: 16px; }}
.card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
         padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.card header {{ display: flex; justify-content: space-between; align-items: center;
                margin: 0 0 8px; }}
.card h2 {{ margin: 0; font-size: 1.1em; text-transform: capitalize; }}
.card img {{ width: 100%; height: auto; max-height: 320px; object-fit: contain;
             background: #222; border-radius: 4px; }}
.meta {{ font-size: 0.85em; color: #444; margin-top: 6px;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.links {{ margin-top: 8px; font-size: 0.85em; }}
.links a {{ margin-right: 6px; color: #06c; text-decoration: none; }}
.links a:hover {{ text-decoration: underline; }}
.pill {{ font-size: 0.7em; padding: 2px 8px; border-radius: 10px;
         text-transform: uppercase; letter-spacing: 0.5px; }}
.pill-ok   {{ background: #d4f5d4; color: #0a6b0a; }}
.pill-fail {{ background: #fde0e0; color: #a12222; }}
.noimg {{ padding: 40px; background: #eee; text-align: center; color: #888;
          border-radius: 4px; }}
code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 3px;
        font-size: 0.88em; }}
</style>
</head><body>
<header class='page'>
  <h1>vlm_wrapper visual grounding run</h1>
  <div class='sub'>{ts} · model={summaries[0].get("model", "?") if summaries else "?"}</div>
</header>
<div class='grid'>
{"".join(cards)}
</div>
</body></html>
"""
    (run_dir / "index.html").write_text(html, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────

def _run_all(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = Path(args.capture_dir)

    summaries: list[dict[str, Any]] = []
    for case in args.cases:
        if case not in tvp._CASE_RUNNERS:
            logger.warning("Unknown case %r — skipping", case)
            continue

        print(f"\n==== Running case: {case} ====", flush=True)
        runner = tvp._CASE_RUNNERS[case]
        t0 = time.perf_counter()
        try:
            case_result, schema = runner(args)
            case_result["case"] = case
            error = None
        except Exception as exc:
            logger.exception("case %s failed", case)
            case_result, schema, error = (
                {"case": case, "error": str(exc)}, None, str(exc)
            )
        elapsed = time.perf_counter() - t0
        case_result["elapsed_s"] = round(elapsed, 2)

        case_dir = run_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)

        # Write raw records first — even on schema failure we keep the
        # trace so a user can diagnose.
        (case_dir / "case.json").write_text(
            json.dumps(case_result, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        (case_dir / "tool_trace.json").write_text(
            json.dumps(case_result.get("tool_trace") or [],
                       indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        if schema:
            (case_dir / "schema.txt").write_text(schema, encoding="utf-8")

        # Copy the input image into the case folder.
        input_dst = case_dir / "input.png"
        copied = False
        if case in ("gymv", "browser", "desktop"):
            src = _case_input_path(case, capture_dir)
            if src:
                shutil.copy(src, input_dst)
                copied = True
        elif case == "tir_bench":
            copied = _copy_tir_bench_input(case_result, input_dst)
        elif case == "video_holmes":
            copied = _copy_video_first_frame(case_result, input_dst)

        # Render annotated overlay if we have both an image and a schema.
        has_annot = False
        if copied and schema:
            try:
                img = Image.open(input_dst).convert("RGB")
                ents = _parse_entities(schema)
                if ents:
                    annot = _annotate(img, ents, case)
                    annot.save(case_dir / "annotated.png")
                    has_annot = True
            except Exception as exc:
                logger.warning("annotation failed for %s: %s", case, exc)

        # Human-readable per-case report.
        (case_dir / "trace_summary.md").write_text(
            _render_case_md(case, case_result, schema, has_annot),
            encoding="utf-8",
        )

        # Scorecard line.
        val = case_result.get("validation") or {}
        status = "OK" if val.get("valid") else "FAIL"
        correct = (
            f" correct={case_result['correct']}"
            if "correct" in case_result else ""
        )
        trace_n = len(case_result.get("tool_trace") or [])
        print(
            f"  [{status}] {case:<14}{elapsed:6.1f}s | "
            f"head={case_result.get('head_used','?')} "
            f"rounds={case_result.get('rounds','?')} "
            f"tool_calls={trace_n} entities={val.get('entity_count',0)}{correct}"
        )
        if error:
            print(f"  ERROR: {error}")

        summaries.append(case_result)

    return summaries


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run tool-calling visual grounding across all "
                    "vlm_wrapper domains and bundle the visual/textual "
                    "artifacts into an HTML gallery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cases", nargs="+",
        default=list(tvp.CASES),
        help="Subset of cases to run (default: all five).",
    )
    p.add_argument(
        "--run-name", default=None,
        help="Explicit bundle directory name (default: timestamp).",
    )
    p.add_argument(
        "--root-dir",
        default=str(Path(tvp.DEFAULT_OUT_DIR).parent / "grounding_runs"),
        help="Parent directory for run bundles.",
    )
    p.add_argument("--capture-dir", default=str(tvp.DEFAULT_CAPTURE_DIR))
    p.add_argument("--out-dir", default=str(tvp.DEFAULT_OUT_DIR))  # passed through
    p.add_argument("--model", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--head", choices=tvp.HEAD_CHOICES, default="auto")
    p.add_argument("--gymv-head",    choices=tvp.HEAD_CHOICES, default=None)
    p.add_argument("--browser-head", choices=tvp.HEAD_CHOICES, default=None)
    p.add_argument("--desktop-head", choices=tvp.HEAD_CHOICES, default=None)

    args = p.parse_args(argv)

    # Match the per-case override defaulting that test_vlm_parsers does.
    args.gymv_head = args.gymv_head or args.head
    args.browser_head = args.browser_head or args.head
    args.desktop_head = args.desktop_head or args.head

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.root_dir) / run_name

    print(f"Run bundle: {run_dir}")
    summaries = _run_all(args, run_dir)

    (run_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    _render_index_html(run_dir, summaries)

    okc = sum(
        1 for s in summaries
        if (s.get("validation") or {}).get("valid")
    )
    print(
        f"\n==== Summary ====\n"
        f"{okc}/{len(summaries)} cases produced a valid <state> schema.\n"
        f"Gallery: {run_dir / 'index.html'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
