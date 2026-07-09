#!/usr/bin/env python3
"""5-task schema_gen smoke (vLLM offline) + base-vs-LoRA-vs-GPT-gold compare.

Picks one held-out sample per task and runs Qwen3.5-35B-A3B (with or without
the schema_gen LoRA) through vLLM offline mode. Targets:

  1. gymv         (any Temporal_*-v0 environment)
  2. env_wrappers / candy_crush
  3. env_wrappers / super_mario
  4. env_wrappers / tetris
  5. env_wrappers / twenty_forty_eight

Modes (``--lora-mode``):
  - ``lora``  (default): 35B + schema_gen LoRA  (T1.1' verification path)
  - ``base``           : 35B base only          (does base alone match gold?)
  - ``both``           : run both back-to-back on the same picks and emit a
                          side-by-side report.  This is the apples-to-apples
                          ablation against the GPT-5.4 gold (also produced
                          single-shot via gymv_wrapper.adapter.generate_label).

Why vLLM and not transformers?
  Sequential decoding on a 35B-A3B MoE with transformers + ``device_map="auto"``
  is dominated by inter-GPU comms and expert-routing overhead — empirically
  ~7 min / sample. vLLM's continuous batching and CUDA graph capture brings
  this to under a minute / sample on a single H200.

Outputs:
  - Per-task table:  domain, task, mode, exact_match, prefix%, n_fields, ...
  - Optional JSON report via ``--report <path>``  (includes full predicted
    text for both legs so you can manually diff against gold)

Usage:
    # Run LoRA only (smoke):
    python evaluation/smoke_schema_gen_5tasks.py
    # Run base only:
    python evaluation/smoke_schema_gen_5tasks.py --lora-mode base
    # Side-by-side compare:
    python evaluation/smoke_schema_gen_5tasks.py --lora-mode both \\
        --report runs/t1_1prime/compare_base_vs_lora.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


_WS_RE = re.compile(r"[ \t]+\n")
_BLANKS_RE = re.compile(r"\n{3,}")
_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)

# Top-level sub-tags emitted inside <state>...</state> by the canonical
# schema_gen training pipeline (see vlm_wrapper.schema._SECTION_MAP).
# We do NOT use the legacy intention/observations/ambiguities triple from
# probe_schema_gen_exact_match.py — that list predates the current spec
# and would mark the model down for *not* hallucinating fields it was
# never trained to produce.
EXPECTED_SCHEMA_FIELDS = (
    "entities",
    "relations",
    "state_flags",
    "targets",
    "uncertainty",
    "actions",
    "evidence",
)


def _canon(text: str) -> str:
    text = _WS_RE.sub("\n", text.strip())
    text = _BLANKS_RE.sub("\n\n", text)
    return text


_TAG_OPEN_RE = re.compile(r"<(\w+)>")
_STATE_RE = re.compile(r"<state>(.*?)(?:</state>|\Z)", re.DOTALL)
# Section markers are always on their own line in the canonical format:
#   ...
#   e7[type=text, label=Timer 99, ...]
#   <state_flags>            ← marker = "<name>" alone on a line
#   round=1
#   ...
_SECTION_MARKER_RE = re.compile(r"(?m)^[ \t]*<(\w+)>[ \t]*$")


def _extract_tags(schema: str) -> Dict[str, str]:
    """Return ``{section_name: section_body}`` for the canonical schema_gen
    format.

    The format uses **section markers**, not paired open/close tags: a line
    like ``<entities>`` opens the entities section, which runs until the next
    section marker or the closing ``</state>``. There are NO ``</entities>``
    / ``</state_flags>`` etc. tags anywhere in the corpus. The pre-amble
    between ``<state>`` and the first marker (``domain=`` / ``task=`` /
    ``goal=`` / ``step=``) is returned under the synthetic key ``"_header"``.
    """
    if not schema:
        return {}
    m = _STATE_RE.search(schema)
    body = m.group(1) if m else schema
    markers = list(_SECTION_MARKER_RE.finditer(body))
    out: Dict[str, str] = {}
    if not markers:
        out["_header"] = body.strip()
        return out
    out["_header"] = body[: markers[0].start()].strip()
    for i, mk in enumerate(markers):
        name = mk.group(1)
        s = mk.end()
        e = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        out[name] = body[s:e].strip()
    return out


def _present_tag_names(schema: str) -> set:
    """All distinct section-marker names that appear (open marker only).
    Tolerates truncated outputs where ``</state>`` never arrived."""
    return set(_TAG_OPEN_RE.findall(schema or ""))


def _path_a_accepts(predicted: str) -> bool:
    if "<state>" not in predicted or "</state>" not in predicted:
        return False
    tags = _extract_tags(predicted)
    n = sum(1 for f in EXPECTED_SCHEMA_FIELDS if f in tags)
    return n >= 3


@dataclass
class TaskPick:
    """One held-out sample picked deterministically for one task."""
    task_id: str
    domain: str
    sample: Any


def _pick_one_per_task(seed: int) -> List[TaskPick]:
    """Sample exactly one held-out triple per (domain, task)."""
    from trainer.SFT.schema_gen.config import SchemaGenConfig
    from trainer.SFT.schema_gen.data_loader import load_schema_gen_dataset

    cfg = SchemaGenConfig(
        domains=["gymv", "env_wrappers"],
        gymv_triple_root=str(REPO_ROOT / "labeling/output/grounding/gymv"),
        env_wrappers_triple_root=str(
            REPO_ROOT / "labeling/output/grounding/env_wrappers"
        ),
        drop_hard_cases=False,
        max_samples_per_domain=None,
    )
    pool = load_schema_gen_dataset(cfg)
    if not pool:
        raise SystemExit("Empty schema_gen pool — check labeling output paths.")

    target_tasks: List[Tuple[str, str]] = [
        ("env_wrappers", "candy_crush"),
        ("env_wrappers", "super_mario"),
        ("env_wrappers", "tetris"),
        ("env_wrappers", "twenty_forty_eight"),
        ("gymv", ""),
    ]

    rng = random.Random(seed)
    picks: List[TaskPick] = []
    for dom, key in target_tasks:
        bucket = [
            s for s in pool
            if s.domain == dom and (key == "" or key in s.sample_id)
        ]
        if not bucket:
            print(
                f"  [warn] no held-out sample found for {dom}/{key or 'any'}",
                file=sys.stderr,
            )
            continue
        s = rng.choice(bucket)
        if dom == "env_wrappers":
            task_id = f"env_wrappers/{key}"
        else:
            env_id = s.sample_id.split(".")[1] if "." in s.sample_id else "?"
            task_id = f"gymv/{env_id}"
        picks.append(TaskPick(task_id=task_id, domain=dom, sample=s))
    return picks


# ── Multi-domain picker (cross-corpus, for the all-5-domain matrix) ──
# Below this line, picks are sourced directly from Cold-start-out-* dumps
# instead of the schema_gen data_loader.  Each adapter returns a synthetic
# ``Sample``-like object exposing the four attributes the rest of the
# script touches: ``sample_id``, ``domain``, ``images`` (list[str]),
# ``prompt`` (str user-message), ``target_schema`` (str gold).


@dataclass
class _SyntheticSample:
    sample_id: str
    domain: str
    images: List[str]
    prompt: str
    target_schema: str


def _stem_prompt_from_schema(schema: str) -> str:
    """Reconstruct a short user-prompt from the gold schema's header.

    The training-time user message was typically the task description /
    goal sentence (see vlm_wrapper.schema.build_user_message).  We pull
    those values out of the gold's preamble so the prompt the model sees
    matches what GPT-5.4 saw.
    """
    if not schema or "<state>" not in schema:
        return "Describe the screenshot using the canonical schema."
    pre = schema.split("<entities>", 1)[0]
    goal = ""
    task = ""
    for line in pre.splitlines():
        if line.startswith("goal="):
            goal = line[5:].strip()
        elif line.startswith("task="):
            task = line[5:].strip()
    parts = []
    if task:
        parts.append(f"Task: {task}")
    if goal:
        parts.append(f"Goal: {goal}")
    parts.append(
        "Output the canonical <state>...</state> schema for the "
        "screenshot above."
    )
    return "\n".join(parts)


def _pick_browser_one(seed: int) -> Optional[TaskPick]:
    """Pick one browser step (Cold-start-out-browsergym/<task>/frames/...)."""
    import glob
    rng = random.Random(seed + 11)
    candidates = sorted(
        glob.glob("Cold-start-out-browsergym/*/frames/ep_*/step_*.json"),
        key=str,
    )
    if not candidates:
        return None
    rng.shuffle(candidates)
    for f in candidates[:200]:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        schema = d.get("schema") or ""
        frame = d.get("frame_path") or ""
        if not schema or "<state>" not in schema or "</state>" not in schema:
            continue
        if not frame:
            continue
        frame_abs = (REPO_ROOT / frame) if not Path(frame).is_absolute() else Path(frame)
        if not frame_abs.exists():
            continue
        first = schema.split("\n", 2)
        if len(first) < 2 or first[1] != "domain=browser":
            continue
        task_dir = Path(f).parts[1]  # Cold-start-out-browsergym/<task_dir>/...
        s = _SyntheticSample(
            sample_id=f"browser.{task_dir}.s{d.get('step', 0)}",
            domain="browser",
            images=[str(frame_abs)],
            prompt=_stem_prompt_from_schema(schema),
            target_schema=schema,
        )
        return TaskPick(task_id=f"browser/{task_dir}", domain="browser", sample=s)
    return None


def _pick_desktop_one(seed: int) -> Optional[TaskPick]:
    """Pick one OSWorld step (filtered to schema-tagged ``domain=desktop``)."""
    import glob
    rng = random.Random(seed + 22)
    candidates = sorted(
        glob.glob(
            "Cold-start-out-osworld/**/frames/ep_*/step_*.json", recursive=True,
        ),
        key=str,
    )
    if not candidates:
        return None
    rng.shuffle(candidates)
    for f in candidates[:300]:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        schema = d.get("schema") or ""
        frame = d.get("frame_path") or ""
        if not schema or "<state>" not in schema or "</state>" not in schema:
            continue
        if not frame:
            continue
        frame_abs = (REPO_ROOT / frame) if not Path(frame).is_absolute() else Path(frame)
        if not frame_abs.exists():
            continue
        first = schema.split("\n", 2)
        if len(first) < 2 or first[1] != "domain=desktop":
            continue
        # OSWorld path: Cold-start-out-osworld/<run>/<app>/<task_uuid>/frames/...
        parts = Path(f).parts
        app = parts[-5] if len(parts) >= 6 else "?"
        task_uuid = parts[-4] if len(parts) >= 5 else "?"
        s = _SyntheticSample(
            sample_id=f"desktop.{app}.{task_uuid}.s{d.get('step', 0)}",
            domain="desktop",
            images=[str(frame_abs)],
            prompt=_stem_prompt_from_schema(schema),
            target_schema=schema,
        )
        return TaskPick(task_id=f"desktop/{app}", domain="desktop", sample=s)
    return None


def _pick_image_qa_one(seed: int, scratch: Path) -> Optional[TaskPick]:
    """Pick one image_qa sample; materialise the HF image to disk."""
    rng = random.Random(seed + 33)
    candidate_files = list(REPO_ROOT.glob(
        "Cold-start-out-visual-reasoning/**/visual_toolbench/samples.jsonl",
    )) + list(REPO_ROOT.glob(
        "Cold-start-out-visual-reasoning/**/tir_bench/samples.jsonl",
    ))
    rows: List[Tuple[str, dict]] = []
    for f in candidate_files:
        with f.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if not d.get("schema"):
                    continue
                first = d["schema"].split("\n", 2)
                if len(first) < 2 or first[1] != "domain=image_qa":
                    continue
                rows.append((d.get("benchmark", "?"), d))
    if not rows:
        return None
    rng.shuffle(rows)
    for bench, d in rows[:30]:
        try:
            if bench == "visual_toolbench":
                from visual_reasoning_wrapper.benchmarks.visual_toolbench import (
                    iter_visual_toolbench_samples,
                    load_visual_toolbench_image,
                )
                # Find the matching iterator entry by sample_id, then load image
                target_sid = d.get("sample_id")
                pil = None
                for s in iter_visual_toolbench_samples(
                    limit=1000, single_turn_only=True,
                ):
                    if getattr(s, "sample_id", None) == target_sid:
                        pil = load_visual_toolbench_image(s)
                        break
                if pil is None:
                    continue
            elif bench == "tir_bench":
                from visual_reasoning_wrapper.benchmarks.tir_bench import (
                    iter_tir_bench_samples,
                    load_tir_bench_image,
                )
                target_sid = d.get("sample_id")
                pil = None
                for s in iter_tir_bench_samples(limit=500):
                    if getattr(s, "sample_id", None) == target_sid:
                        pil = load_tir_bench_image(s)
                        break
                if pil is None:
                    continue
            else:
                continue
        except Exception as exc:
            print(
                f"  [warn] image_qa loader failed for {bench}/"
                f"{d.get('sample_id')}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        scratch.mkdir(parents=True, exist_ok=True)
        img_path = scratch / f"image_qa_{bench}_{d.get('sample_id')}.png"
        try:
            pil.convert("RGB").save(img_path, format="PNG")
        except Exception as exc:
            print(
                f"  [warn] failed to save image_qa PNG: {exc}", file=sys.stderr,
            )
            continue
        s = _SyntheticSample(
            sample_id=f"image_qa.{bench}.{d.get('sample_id')}",
            domain="image_qa",
            images=[str(img_path)],
            prompt=_stem_prompt_from_schema(d["schema"]),
            target_schema=d["schema"],
        )
        return TaskPick(
            task_id=f"image_qa/{bench}", domain="image_qa", sample=s,
        )
    return None


def _pick_video_qa_one(seed: int, scratch: Path) -> Optional[TaskPick]:
    """Pick one video_qa sample; extract the labeler-indexed frames."""
    rng = random.Random(seed + 44)
    candidate_files = [
        REPO_ROOT / "Cold-start-out-visual-reasoning-video/siv_bench/samples.jsonl",
        REPO_ROOT / "Cold-start-out-visual-reasoning-video/video_holmes/samples.jsonl",
    ]
    rows: List[dict] = []
    for f in candidate_files:
        if not f.is_file():
            continue
        with f.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if not d.get("schema"):
                    continue
                first = d["schema"].split("\n", 2)
                if len(first) < 2 or first[1] != "domain=video_qa":
                    continue
                vm = d.get("video_meta") or {}
                vp = vm.get("video_path")
                if not vp or not Path(vp).exists():
                    continue
                rows.append(d)
    if not rows:
        return None
    rng.shuffle(rows)
    for d in rows[:10]:
        vm = d["video_meta"]
        vp = vm["video_path"]
        indices = vm.get("indices") or []
        if not indices:
            continue
        try:
            import decord  # type: ignore
            # Cap to a small set of evenly-spaced frames so we stay well
            # within the vLLM context budget (each frame is ~1k–4k image
            # tokens after Qwen2.5-VL tiling).  4 frames is enough for the
            # smoke; the GPT-5.4 labeler used 8.
            max_frames = 4
            if len(indices) > max_frames:
                step_idx = max(1, len(indices) // max_frames)
                indices = indices[::step_idx][:max_frames]
            vr = decord.VideoReader(vp)
            frames_arr = vr.get_batch(indices).asnumpy()
            from PIL import Image as PILImage
            scratch.mkdir(parents=True, exist_ok=True)
            saved: List[str] = []
            sid = d.get("sample_id", "video")
            stem = scratch / f"video_qa_{d.get('benchmark', '?')}_{sid}"
            for i, arr in enumerate(frames_arr):
                p = Path(f"{stem}_f{i:02d}.png")
                # Downscale long edge to 768 px to keep image tokens modest.
                im = PILImage.fromarray(arr)
                long_edge = max(im.size)
                if long_edge > 768:
                    scale = 768 / long_edge
                    im = im.resize(
                        (int(im.size[0] * scale), int(im.size[1] * scale)),
                        PILImage.BILINEAR,
                    )
                im.save(p, format="PNG")
                saved.append(str(p))
        except Exception as exc:
            print(
                f"  [warn] video frame extract failed for {sid}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        s = _SyntheticSample(
            sample_id=f"video_qa.{d.get('benchmark', '?')}.{sid}",
            domain="video_qa",
            images=saved,
            prompt=_stem_prompt_from_schema(d["schema"]),
            target_schema=d["schema"],
        )
        return TaskPick(
            task_id=f"video_qa/{d.get('benchmark', '?')}",
            domain="video_qa", sample=s,
        )
    return None


def _pick_one_per_domain_5(seed: int, scratch: Path) -> List[TaskPick]:
    """Pick one (held-out) sample for each of the 5 user-listed domains.

    Returns up to 5 picks: games (gymv OR env_wrappers), browser, desktop,
    image_qa, video_qa.  Skips a domain if no usable sample is on disk.
    """
    picks: List[TaskPick] = []
    games = _pick_one_per_task(seed=seed)
    if games:
        picks.append(games[0])  # one games-domain representative
    for fn in (_pick_browser_one, _pick_desktop_one):
        p = fn(seed)
        if p:
            picks.append(p)
        else:
            print(f"  [warn] no usable {fn.__name__} pick", file=sys.stderr)
    for fn in (_pick_image_qa_one, _pick_video_qa_one):
        p = fn(seed, scratch)
        if p:
            picks.append(p)
        else:
            print(f"  [warn] no usable {fn.__name__} pick", file=sys.stderr)
    return picks


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@dataclass
class TaskResult:
    task_id: str
    domain: str
    sample_id: str
    image_path: str
    mode: str = "lora"
    elapsed_s: float = 0.0
    n_predicted_chars: int = 0
    n_gold_chars: int = 0
    common_prefix_chars: int = 0
    common_prefix_pct: float = 0.0
    exact_match: bool = False
    path_a_accept: bool = False
    state_block_closed: bool = False
    pred_tag_names: List[str] = field(default_factory=list)
    gold_tag_names: List[str] = field(default_factory=list)
    n_canonical_fields_present: int = 0
    field_overlap: Dict[str, bool] = field(default_factory=dict)
    predicted_full: str = ""
    gold_full: str = ""
    error: Optional[str] = None


def _score_one(
    sample: Any,
    predicted: str,
    elapsed_s: float,
    task_id: str,
    *,
    mode: str = "lora",
) -> TaskResult:
    gold = _canon(sample.target_schema or "")
    pred = _canon(predicted or "")
    em = bool(gold) and gold == pred

    gold_tags = _extract_tags(gold)
    pred_tags = _extract_tags(pred)
    pred_present = _present_tag_names(pred)
    gold_present = _present_tag_names(gold)
    field_overlap = {
        f: (f in gold_tags and f in pred_tags and gold_tags[f] == pred_tags[f])
        for f in EXPECTED_SCHEMA_FIELDS
    }
    n_present = sum(1 for f in EXPECTED_SCHEMA_FIELDS if f in pred_present)
    cpl = _common_prefix_len(gold, pred)

    return TaskResult(
        task_id=task_id,
        domain=sample.domain,
        sample_id=sample.sample_id,
        image_path=str(sample.images[0]) if sample.images else "",
        mode=mode,
        elapsed_s=elapsed_s,
        n_predicted_chars=len(predicted or ""),
        n_gold_chars=len(gold),
        common_prefix_chars=cpl,
        common_prefix_pct=(cpl / max(1, len(gold))),
        exact_match=em,
        path_a_accept=_path_a_accepts(pred),
        state_block_closed=("<state>" in pred and "</state>" in pred),
        pred_tag_names=sorted(pred_present),
        gold_tag_names=sorted(gold_present),
        n_canonical_fields_present=n_present,
        field_overlap=field_overlap,
        predicted_full=pred,
        gold_full=gold,
    )


def _build_messages(
    sample: Any,
    *,
    n_shot: int = 0,
    task_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build the single-shot chat messages exactly as the schema_gen training
    pipeline does (system = adaptive prompt, user = images + sample.prompt).

    This is the apples-to-apples shape — the GPT-5.4 gold was also produced
    by sending these same images + this same domain prompt to a vision LLM
    via gymv_wrapper.adapter.generate_label (single-shot, no tool-calling).
    Any difference in scores between modes is therefore attributable to the
    model identity (GPT-5.4 vs Qwen3.5-35B-A3B), the LoRA, or — when
    ``n_shot > 0`` — the few-shot example block in the system prompt.
    """
    from vlm_wrapper.schema import build_adaptive_system_prompt
    from vlm_wrapper.few_shot_library import get_few_shot_examples

    # ``env_wrappers`` shares the gymv schema spec — fall back to gymv if the
    # few-shot library has no env_wrappers-specific entry yet.  When
    # ``task_id`` is provided, the loader prefers ``{domain}.{task_slug}.txt``
    # over the generic ``{domain}.txt`` (avoids 2048 vocabulary leaking into
    # candy_crush prompts).
    examples = get_few_shot_examples(
        sample.domain, n=n_shot,
        task_id=task_id,
        fallback_domain="gymv" if sample.domain == "env_wrappers" else None,
    )
    sys_prompt = build_adaptive_system_prompt(
        domain=sample.domain, max_entities=25,
        few_shot_examples=examples or None,
    )
    user_content: List[Dict[str, Any]] = []
    for img_path in sample.images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"file://{Path(img_path).resolve()}",
            },
        })
    user_content.append({"type": "text", "text": sample.prompt})
    return [
        {"role": "system",
         "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": user_content},
    ]


def _run_one_pick(
    llm: Any,
    sample: Any,
    sampling_params: Any,
    lora_request: Any,
    task_id: str,
    mode: str,
    *,
    n_shot: int = 0,
) -> TaskResult:
    """Generate one schema for one (sample, mode) and score it against gold."""
    s0 = time.monotonic()
    try:
        # Pass the task_id through so the few-shot library can route to
        # task-specific examples (e.g. env_wrappers/candy_crush ->
        # env_wrappers.candy_crush.txt) when available.
        messages = _build_messages(
            sample, n_shot=n_shot, task_id=task_id,
        )
        # Qwen3.5's default chat template enables a <think> CoT prefix.
        # The schema_gen LoRA was trained without thinking blocks, and base
        # 35B's CoT will fill max_tokens with reasoning before the schema.
        # Disabling thinking is required for both legs to be comparable.
        out = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        predicted = out[0].outputs[0].text if out and out[0].outputs else ""
        return _score_one(
            sample, predicted, time.monotonic() - s0, task_id, mode=mode,
        )
    except Exception as exc:                       # noqa: BLE001
        return TaskResult(
            task_id=task_id,
            domain=sample.domain,
            sample_id=sample.sample_id,
            image_path=str(sample.images[0]) if sample.images else "",
            mode=mode,
            elapsed_s=time.monotonic() - s0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _resolve_adapter(adapter_arg: Optional[str]) -> Path:
    if adapter_arg:
        return Path(adapter_arg).resolve()
    sg = REPO_ROOT / "runs" / "sft_schema_gen"
    candidates = sorted(
        (d for d in sg.iterdir() if d.is_dir() and d.name.startswith("schema_gen_")),
        key=lambda d: d.name,
        reverse=True,
    )
    for c in candidates:
        if (c / "adapter_config.json").is_file() and (
            c / "adapter_model.safetensors"
        ).is_file():
            return c
    raise SystemExit(f"No loadable schema_gen adapter under {sg}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="5-task schema_gen smoke (vLLM offline + LoRA).",
    )
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--lora-mode", choices=["lora", "base", "both"], default="lora",
        help=(
            "lora=schema_gen LoRA only (default), base=Qwen3.5-35B base only, "
            "both=run base and lora on the same picks for side-by-side diff "
            "vs the GPT-5.4 gold."
        ),
    )
    parser.add_argument(
        "--n-shot", type=int, default=0,
        help=(
            "Number of curated few-shot example schemas to inject in the "
            "system prompt (per pick).  0 = current behaviour (no examples)."
        ),
    )
    parser.add_argument(
        "--shot-grid", type=str, default=None,
        help=(
            "Comma-separated shot counts to sweep, e.g. '0,1,2'.  When set, "
            "each pick is generated once per (mode, n_shot) combination."
        ),
    )
    parser.add_argument(
        "--cover-all-domains", action="store_true",
        help=(
            "Pick one held-out sample for each of the 5 user-listed domains "
            "(games, browser, desktop, image_qa, video_qa) instead of the "
            "default 5-game-tasks set.  Requires Cold-start-out-* dumps on "
            "disk for browser/desktop/image_qa/video_qa."
        ),
    )
    parser.add_argument(
        "--scratch-dir", type=Path,
        default=REPO_ROOT / "runs" / "t1_1prime" / "smoke_scratch",
        help=(
            "Where to save materialised image_qa images and extracted "
            "video_qa frames (used only with --cover-all-domains)."
        ),
    )
    parser.add_argument(
        "--gpu-mem-util", type=float, default=0.85,
        help="vLLM gpu_memory_utilization (default 0.85, leaves headroom).",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HOME", "/workspace/huggingface")
    os.environ.setdefault(
        "HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"),
    )
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    adapter_path = _resolve_adapter(args.adapter)
    cfg = json.loads((adapter_path / "adapter_config.json").read_text())
    base_model = cfg.get("base_model_name_or_path")
    lora_rank = int(cfg.get("r", 16))
    print(
        f"\n=== schema_gen 5-task smoke ===\n"
        f"  adapter   = {adapter_path}\n"
        f"  base      = {base_model}\n"
        f"  lora_r    = {lora_rank}\n"
        f"  max_tokens= {args.max_tokens}\n"
        f"  seed      = {args.seed}\n",
        file=sys.stderr,
    )

    if args.cover_all_domains:
        picks = _pick_one_per_domain_5(
            seed=args.seed, scratch=args.scratch_dir,
        )
    else:
        picks = _pick_one_per_task(seed=args.seed)
    print(f"Picked {len(picks)} task(s):", file=sys.stderr)
    for p in picks:
        print(
            f"  - {p.task_id:<35s}  "
            f"sample_id={p.sample.sample_id}  "
            f"images={len(p.sample.images)}",
            file=sys.stderr,
        )

    from PIL import Image  # noqa: F401  (forces wheel resolution)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t_init0 = time.monotonic()
    llm = LLM(
        model=base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_lora=True,
        max_lora_rank=max(16, lora_rank),
        max_loras=1,
        max_model_len=16384,
        enforce_eager=False,
        disable_log_stats=True,
        allowed_local_media_path=str(REPO_ROOT),
    )
    print(
        f"\nvLLM ready in {time.monotonic() - t_init0:.1f}s\n",
        file=sys.stderr,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=None,
    )
    lora_req = LoRARequest("schema_gen", 1, str(adapter_path))

    if args.lora_mode == "lora":
        modes: List[str] = ["lora"]
    elif args.lora_mode == "base":
        modes = ["base"]
    else:
        modes = ["base", "lora"]

    if args.shot_grid:
        try:
            shot_counts = sorted({
                int(x.strip()) for x in args.shot_grid.split(",") if x.strip()
            })
        except ValueError:
            raise SystemExit(
                f"--shot-grid must be comma-separated ints; got "
                f"{args.shot_grid!r}"
            )
    else:
        shot_counts = [args.n_shot]

    print(
        f"\nLora mode = {args.lora_mode}  →  modes to run: {modes}\n"
        f"Shot counts to sweep: {shot_counts}",
        file=sys.stderr,
    )

    results: List[TaskResult] = []
    for p in picks:
        s = p.sample
        for mode in modes:
            req = lora_req if mode == "lora" else None
            for n_shot in shot_counts:
                row = _run_one_pick(
                    llm=llm,
                    sample=s,
                    sampling_params=sampling_params,
                    lora_request=req,
                    task_id=p.task_id,
                    mode=mode,
                    n_shot=n_shot,
                )
                # Carry the n_shot label inside the mode tag so downstream
                # aggregation can group by leg without changing TaskResult.
                row.mode = f"{mode}+{n_shot}shot"
                results.append(row)
                if row.exact_match:
                    marker = "EM"
                elif row.state_block_closed and row.common_prefix_pct >= 0.5:
                    marker = "OK"
                elif row.state_block_closed:
                    marker = "ok"
                elif row.error:
                    marker = "ER"
                else:
                    marker = "x "
                print(
                    f"  [{marker}] {row.task_id:<35s} mode={row.mode:<14s} "
                    f"prefix={row.common_prefix_chars:>5d}/"
                    f"{row.n_gold_chars:>5d} "
                    f"({row.common_prefix_pct*100:>5.1f}%)  "
                    f"fields={row.n_canonical_fields_present}/"
                    f"{len(EXPECTED_SCHEMA_FIELDS)}  "
                    f"chars={row.n_predicted_chars:>5d}  "
                    f"{row.elapsed_s:>6.1f}s"
                    + (f"  ERROR={row.error}" if row.error else ""),
                    file=sys.stderr,
                )

    def _agg(rs: List[TaskResult]) -> Dict[str, Any]:
        n = max(1, len(rs))
        return {
            "n": len(rs),
            "exact_match": sum(1 for r in rs if r.exact_match),
            "state_block_closed": sum(1 for r in rs if r.state_block_closed),
            "path_a_accept": sum(1 for r in rs if r.path_a_accept),
            "mean_prefix_pct": sum(r.common_prefix_pct for r in rs) / n,
            "mean_fields": sum(r.n_canonical_fields_present for r in rs) / n,
            "mean_chars": sum(r.n_predicted_chars for r in rs) / n,
            "mean_elapsed_s": sum(r.elapsed_s for r in rs) / n,
            "field_accuracy": {
                f: (sum(1 for r in rs if r.field_overlap.get(f)) / n)
                for f in EXPECTED_SCHEMA_FIELDS
            },
        }

    by_mode: Dict[str, List[TaskResult]] = {}
    for r in results:
        by_mode.setdefault(r.mode, []).append(r)

    summaries = {m: _agg(rs) for m, rs in by_mode.items()}

    print("\n=== per-mode summary ===", file=sys.stderr)
    for mode, agg in summaries.items():
        print(
            f"\n[mode={mode}]"
            f"  n={agg['n']}  EM={agg['exact_match']}/{agg['n']}"
            f"  </state>={agg['state_block_closed']}/{agg['n']}"
            f"  pathA={agg['path_a_accept']}/{agg['n']}"
            f"  mean_prefix={agg['mean_prefix_pct']*100:.1f}%"
            f"  mean_fields={agg['mean_fields']:.2f}/{len(EXPECTED_SCHEMA_FIELDS)}"
            f"  mean_chars={agg['mean_chars']:.0f}"
            f"  mean_t={agg['mean_elapsed_s']:.1f}s",
            file=sys.stderr,
        )
        print(
            "  per-field exact-match (pred vs gold tag-body):\n" +
            "".join(
                f"    - {k:<14s} {v:.2f}\n"
                for k, v in agg["field_accuracy"].items()
            ),
            file=sys.stderr,
        )

    # Side-by-side diff vs gold across all legs (mode × n_shot)
    leg_names = list(by_mode.keys())
    if len(leg_names) >= 2:
        # Per-task wide table: one column per leg
        col_w = 10
        print(
            "\n=== side-by-side per-task common-prefix % vs GPT-5.4 gold ===\n"
            + f"  {'task':<40s}  {'gold_chars':>10s}"
            + "".join(f"  {n:>{col_w}s}" for n in leg_names),
            file=sys.stderr,
        )
        # build sample_id → {leg → row}
        per_sid: Dict[str, Dict[str, TaskResult]] = {}
        for n, rs in by_mode.items():
            for r in rs:
                per_sid.setdefault(r.sample_id, {})[n] = r
        for sid in sorted(per_sid):
            row_by_leg = per_sid[sid]
            ref = next(iter(row_by_leg.values()))
            line = f"  {ref.task_id:<40s}  {ref.n_gold_chars:>10d}"
            for n in leg_names:
                rr = row_by_leg.get(n)
                if rr is None:
                    line += f"  {'-':>{col_w}s}"
                else:
                    line += f"  {rr.common_prefix_pct*100:>{col_w-1}.1f}%"
            print(line, file=sys.stderr)

        # Per-task fields-present table
        print(
            "\n=== side-by-side per-task canonical-sections-present (n/6) ==="
            + f"\n  {'task':<40s}"
            + "".join(f"  {n:>{col_w}s}" for n in leg_names),
            file=sys.stderr,
        )
        for sid in sorted(per_sid):
            row_by_leg = per_sid[sid]
            ref = next(iter(row_by_leg.values()))
            line = f"  {ref.task_id:<40s}"
            for n in leg_names:
                rr = row_by_leg.get(n)
                if rr is None:
                    line += f"  {'-':>{col_w}s}"
                else:
                    line += (
                        f"  {rr.n_canonical_fields_present:>{col_w-2}d}/"
                        f"{len(EXPECTED_SCHEMA_FIELDS)}"
                    )
            print(line, file=sys.stderr)

        # Per-leg aggregates ranked by mean-prefix
        print("\n=== legs ranked by mean common-prefix % vs gold ===",
              file=sys.stderr)
        ranked = sorted(
            summaries.items(),
            key=lambda kv: kv[1]["mean_prefix_pct"], reverse=True,
        )
        for leg, agg in ranked:
            print(
                f"  {leg:<14s} "
                f"prefix={agg['mean_prefix_pct']*100:5.1f}%  "
                f"fields={agg['mean_fields']:.2f}/{len(EXPECTED_SCHEMA_FIELDS)}  "
                f"</state>={agg['state_block_closed']}/{agg['n']}  "
                f"pathA={agg['path_a_accept']}/{agg['n']}  "
                f"mean_t={agg['mean_elapsed_s']:.1f}s",
                file=sys.stderr,
            )

        # Diagnostic: best base+Nshot prefix vs lora+0shot
        base_legs = [n for n in leg_names if n.startswith("base+")]
        lora_legs = [n for n in leg_names if n.startswith("lora+")]
        if base_legs and lora_legs:
            best_base = max(
                base_legs, key=lambda n: summaries[n]["mean_prefix_pct"],
            )
            best_lora = max(
                lora_legs, key=lambda n: summaries[n]["mean_prefix_pct"],
            )
            d = (summaries[best_lora]["mean_prefix_pct"]
                 - summaries[best_base]["mean_prefix_pct"]) * 100
            print(
                "\n=== headline ===\n"
                f"  best base config  = {best_base:<14s}  "
                f"prefix={summaries[best_base]['mean_prefix_pct']*100:5.1f}%\n"
                f"  best lora config  = {best_lora:<14s}  "
                f"prefix={summaries[best_lora]['mean_prefix_pct']*100:5.1f}%\n"
                f"  Δ (lora − base)   = {d:+.1f} pp\n"
                "\n  Interpretation:\n"
                "    - If Δ ≤ 0 the LoRA is no better than base + few-shot\n"
                "      and you can drop the LoRA for OOD domains.\n"
                "    - If Δ > 0 the LoRA still adds value beyond what\n"
                "      few-shot prompting alone delivers.\n",
                file=sys.stderr,
            )

    print(
        "\nTarget (Phase-1, PLAN-VISUAL-GROUNDING-MILESTONES section 13):\n"
        "  - Path-A acceptance >= 0.70\n"
        "  - field accuracy    >= 0.85 on gymv\n",
        file=sys.stderr,
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "adapter_path": str(adapter_path),
            "base_model": base_model,
            "lora_mode": args.lora_mode,
            "modes_run": list(by_mode.keys()),
            "summaries_by_mode": summaries,
            "results": [asdict(r) for r in results],
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nReport -> {args.report}", file=sys.stderr)

    return 0 if all(not r.error for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
