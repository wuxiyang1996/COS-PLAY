"""SIV-Bench loader + GPT-4o parser.

SIV-Bench (https://kfq20.github.io/sivbench/) is a 2,792-clip /
8,728-MCQ social-interaction video benchmark.  Each question targets
one of three core dimensions — Social Scene Understanding (SSU),
Social State Reasoning (SSR), Social Dynamics Prediction (SDP) — split
into 10 fine-grained sub-tasks.  The dataset ships with three subtitle
conditions (``origin``, ``w_sub``, ``wo_sub``) so the same clip can be
evaluated with / without dialogue text overlays.

Disk layout expected at ``default_siv_bench_root()`` — i.e.
``data/SIV-Bench/`` relative to the repo root::

    <root>/
        SIV-Bench-QA.tsv                # 3.36 MB QA table
        origin/<video_id>.mp4           # original clips
        w_sub/<video_id>.mp4            # +transcribed subtitles
        wo_sub/<video_id>.mp4           # text overlays removed

The TSV column names are detected at load time (case-insensitive) so
the loader works with both the upstream layout and any locally
re-exported version.  Required columns:

* a video identifier (``video_id`` / ``video`` / ``id``)
* a question (``question``)
* a list of options — either a single ``options`` JSON column or
  ``A``…``E`` columns
* a gold ``answer`` letter (A–E)

Usage::

    from visual_reasoning_wrapper.benchmarks.siv_bench import (
        iter_siv_bench_samples, parse_siv_bench_sample,
    )

    for sample in iter_siv_bench_samples(limit=3):
        out = parse_siv_bench_sample(
            sample, num_frames=8, model="gpt-4o", api_key=KEY,
        )
        print(sample.dimension, out["answer"], "gt:", sample.answer)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from PIL import Image

from vlm_wrapper.ground import GroundingRequest, cascaded_ground
from ..question_router import classify_question
from .video_holmes import sample_video_frames

logger = logging.getLogger(__name__)

# Variable MCQ width.  The SIV-Bench *paper* describes 5-way MCQ
# (1 correct + 4 distractors, §B.2), but the released TSV ships some
# sub-tasks (notably "Relation Inference") with up to **12 options**
# (e.g. "A. service, B. grandparent-child, … L. boss-employee").
# Truncating to A..E silently dropped the gold answer for those rows
# (cold-start sweep observed gold='L' against a 5-option set),
# leaving the actor mathematically forced to pick a wrong letter.
# We carry the full A..L letter span end-to-end and let downstream
# code skip rows with too-few or too-many options.
_ANSWER_LETTERS = (
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
)

_VIDEO_SUBDIRS = ("origin", "w_sub", "wo_sub")

# Canonical → set of TSV header aliases (case-insensitive lookup).
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "video_id":   ("video_id", "video", "videoid", "id", "vid"),
    "question":   ("question", "q", "query"),
    "answer":     ("correct_answer_index", "correct_answer", "answer_letter", "gt", "label", "correct", "answer"),
    "options":    ("options", "choices"),
    "dimension":  ("dimension", "category", "core_task", "main_task"),
    "subtask":    ("subtask", "sub_task", "fine_grained_task", "task"),
    "subtitle":   ("subtitle", "subtitle_condition", "sub_cond", "condition"),
    "explanation": ("explanation", "rationale", "reason"),
}


@dataclass
class SIVBenchSample:
    """One SIV-Bench MCQ paired with a video clip on disk.

    ``video_path`` resolves the (subtitle-condition × video_id) tuple to
    a concrete ``.mp4`` on disk; falls back to whichever subtitle
    condition is available if the requested one is missing.
    """

    video_id: str
    question_id: int
    question: str
    options: dict[str, str]
    answer: str | None = None
    dimension: str | None = None
    subtask: str | None = None
    subtitle: str = "origin"
    explanation: str | None = None
    video_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "question_id": self.question_id,
            "question": self.question,
            "options": dict(self.options),
            "answer": self.answer,
            "dimension": self.dimension,
            "subtask": self.subtask,
            "subtitle": self.subtitle,
            "explanation": self.explanation,
            "video_path": str(self.video_path) if self.video_path else None,
        }

    def format_question(self) -> str:
        """Render the MCQ prompt the way SIV-Bench evaluates it.

        Mirrors the Video-Holmes prompt structure so the same
        ``video_qa`` cascade can consume both benchmarks without
        per-benchmark prompt branching.
        """
        lines = [self.question.strip(), "", "Options:"]
        for letter in _ANSWER_LETTERS:
            opt = self.options.get(letter)
            if opt is None:
                continue
            lines.append(f"{letter}. {opt}")
        lines.append("")
        lines.append(
            "Procedure (follow strictly):\n"
            "1. FIRST hop must be temporal — call `find_moment`, "
            "`detect_scene_changes`, or `sample_frames` to localise the "
            "social interaction in the clip before reasoning about "
            "people / objects.  Skipping this leaves <entities> "
            "ungrounded.\n"
            "2. Then call `detect_objects_at_frame` on the localised "
            "frames; for tracked people use `track_object` across the "
            "selected window.  For dialogue / on-screen text use "
            "`read_text_in_frame`.\n"
            "3. In <entities>, list the people / objects / events you "
            "detected, each with `pos=x,y,w,h` (bbox in the sampled "
            "frame) or `null`, and an `ontology=` type.  In <targets>, "
            "`target=` is the entity ID of the social referent the "
            "question is about (NOT the MCQ option letter).  "
            "`candidate_set=[e1,e2,...]` is a list of ENTITY IDs.\n"
            "4. Each <evidence> hop must declare `abstract_op=` "
            "(GROUND / CHECK / RETRIEVE / CONCLUDE) and `tool=` of the "
            "ACTUAL tool you called — do NOT invent hops.  Include "
            "`frame=<idx>` and `timestamp=<s>` for every temporal hop.\n"
            "5. In <state_flags>, set `scene_type=video_segment`, "
            "`progress=null`, `phase=null`, `dialog_open=false`, "
            "`input_pending=false`.\n"
            "6. Inside <answer>, put a SINGLE letter (A-E) on the line "
            "beginning with 'answer=' — for example 'answer=C'."
        )
        return "\n".join(lines)


# ======================================================================
# Disk layout helpers
# ======================================================================

def default_siv_bench_root(
    workspace_root: str | Path | None = None,
) -> Path:
    """Return the canonical SIV-Bench root on this workspace."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
    return Path(workspace_root) / "data" / "SIV-Bench"


def _qa_tsv_path(root: Path) -> Path:
    return root / "SIV-Bench-QA.tsv"


def _resolve_video_path(
    root: Path, video_id: str, preferred_subtitle: str,
) -> tuple[Path | None, str]:
    """Find the .mp4 for ``video_id`` under the preferred subtitle dir.

    Falls back to the other two conditions if the preferred subdir
    doesn't have the clip — useful when a partial download is on disk.
    Returns ``(path, actual_subtitle_used)``.
    """
    order = [preferred_subtitle] + [
        s for s in _VIDEO_SUBDIRS if s != preferred_subtitle
    ]
    for sub in order:
        candidate = root / sub / f"{video_id}.mp4"
        if candidate.exists():
            return candidate, sub
    return None, preferred_subtitle


# ======================================================================
# Header detection
# ======================================================================

def _normalise_header(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _build_field_index(headers: Sequence[str]) -> dict[str, str]:
    """Map canonical field names to the actual header strings.

    The TSV ships with several alternative column-naming conventions
    depending on which release you grab.  We accept any of them by
    walking the alias table.  Unknown headers are passed through to the
    sample's ``raw`` dict so callers can still inspect them.
    """
    norm_to_real: dict[str, str] = {
        _normalise_header(h): h for h in headers
    }
    found: dict[str, str] = {}
    for canonical, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            if alias in norm_to_real:
                found[canonical] = norm_to_real[alias]
                break
    # MCQ options encoded as A / B / C / D / E columns.
    for letter in _ANSWER_LETTERS:
        for variant in (letter, letter.lower(), f"option_{letter.lower()}",
                        f"opt_{letter.lower()}"):
            if variant in norm_to_real:
                found[f"opt_{letter}"] = norm_to_real[variant]
                break
    return found


# ======================================================================
# Loaders
# ======================================================================

def load_siv_bench_questions(
    *,
    siv_root: str | Path | None = None,
    limit: int | None = None,
    subtitle: str = "origin",
) -> list[dict[str, Any]]:
    """Load the raw SIV-Bench QA rows from the TSV.

    Returns a list of dicts (one per question) with keys preserved as
    the TSV's original headers.  Use ``iter_siv_bench_samples`` for the
    typed ``SIVBenchSample`` dataclass and on-disk video resolution.
    """
    root = Path(siv_root) if siv_root else default_siv_bench_root()
    qpath = _qa_tsv_path(root)
    if not qpath.exists():
        raise FileNotFoundError(
            f"SIV-Bench QA TSV not found at {qpath}.  Download via "
            f"`huggingface-cli download Fancylalala/SIV-Bench "
            f"--repo-type dataset --local-dir {root}` and unpack the "
            f"video subdirs."
        )

    rows: list[dict[str, Any]] = []
    with qpath.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_siv_bench_samples(
    *,
    siv_root: str | Path | None = None,
    limit: int | None = None,
    subtitle: str = "origin",
    dimensions: Iterable[str] | None = None,
    subtasks: Iterable[str] | None = None,
    video_ids: Iterable[str] | None = None,
    require_video: bool = True,
) -> Iterator[SIVBenchSample]:
    """Yield ``SIVBenchSample`` objects.

    Parameters
    ----------
    siv_root : path, optional
        Override ``data/SIV-Bench/``.
    limit : int, optional
        Stop after this many samples.
    subtitle : str
        Preferred subtitle condition: ``origin``, ``w_sub``, or
        ``wo_sub``.  Falls back to whichever directory has the clip.
    dimensions : iterable of str, optional
        Restrict to one of ``SSU`` / ``SSR`` / ``SDP``.
    subtasks : iterable of str, optional
        Restrict to specific fine-grained sub-tasks (10 in total — see
        the paper §3 / Figure 5).
    video_ids : iterable of str, optional
        Restrict to a whitelist of video IDs.
    """
    if subtitle not in _VIDEO_SUBDIRS:
        raise ValueError(
            f"subtitle must be one of {_VIDEO_SUBDIRS}, got {subtitle!r}"
        )

    root = Path(siv_root) if siv_root else default_siv_bench_root()
    rows = load_siv_bench_questions(
        siv_root=root, limit=None, subtitle=subtitle,
    )
    if not rows:
        return

    headers = list(rows[0].keys())
    field_idx = _build_field_index(headers)
    if "video_id" not in field_idx or "question" not in field_idx:
        raise RuntimeError(
            "SIV-Bench TSV is missing a recognised video_id / question "
            f"column.  Found headers: {headers!r}"
        )

    dim_set = set(dimensions) if dimensions else None
    sub_set = set(subtasks) if subtasks else None
    vid_set = set(video_ids) if video_ids else None

    count = 0
    for qid, row in enumerate(rows):
        vid = str(row.get(field_idx["video_id"], "")).strip()
        if not vid:
            continue
        if vid_set is not None and vid not in vid_set:
            continue

        dimension = (
            row.get(field_idx["dimension"]) if "dimension" in field_idx else None
        )
        subtask = (
            row.get(field_idx["subtask"]) if "subtask" in field_idx else None
        )
        if dim_set is not None and dimension not in dim_set:
            continue
        if sub_set is not None and subtask not in sub_set:
            continue

        options = _parse_options(row, field_idx)
        if not options:
            # Skip malformed rows rather than crash the iteration —
            # collection scripts can grep the warnings.
            logger.warning(
                "SIV-Bench row %d (video %s) has no parseable options; "
                "skipping.  Row keys: %r",
                qid, vid, list(row.keys()),
            )
            continue

        answer = None
        if "answer" in field_idx:
            raw_ans = str(row[field_idx["answer"]]).strip()
            if raw_ans:
                upper = raw_ans.upper()
                if upper[:1] in _ANSWER_LETTERS:
                    answer = upper[:1]
                elif raw_ans.isdigit():
                    idx = int(raw_ans)
                    if 0 <= idx < len(_ANSWER_LETTERS):
                        answer = _ANSWER_LETTERS[idx]
                    elif 1 <= idx <= len(_ANSWER_LETTERS):
                        answer = _ANSWER_LETTERS[idx - 1]
                    else:
                        answer = upper
                else:
                    answer = upper

        explanation = (
            row.get(field_idx["explanation"])
            if "explanation" in field_idx else None
        )

        # Subtitle column overrides the iter-time preference (so a
        # mixed-condition TSV is possible — most releases ship one
        # subtitle condition per row).
        actual_pref = subtitle
        if "subtitle" in field_idx:
            cell = str(row.get(field_idx["subtitle"], "")).strip().lower()
            if cell in _VIDEO_SUBDIRS:
                actual_pref = cell

        vpath, used_sub = _resolve_video_path(root, vid, actual_pref)
        if require_video and vpath is None:
            continue

        sample = SIVBenchSample(
            video_id=vid,
            question_id=qid,
            question=str(row[field_idx["question"]]).strip(),
            options=options,
            answer=answer,
            dimension=dimension,
            subtask=subtask,
            subtitle=used_sub,
            explanation=explanation,
            video_path=vpath,
            raw=row,
        )
        yield sample
        count += 1
        if limit is not None and count >= limit:
            return


def _parse_options(
    row: dict[str, Any], field_idx: dict[str, str],
) -> dict[str, str]:
    """Extract MCQ options from a row.

    Supports three formats observed across SIV-Bench releases:

    * JSON object: ``{"A": "...", "B": "...", ...}``
    * JSON array:  ``["A. ...", "B. ...", ...]``
    * Flat string: ``"A. ..., B. ..., C. ..., D. ..., E. ..."`` —
      the upstream Hugging Face TSV ships this format inside a single
      ``options`` cell.  We split on letter-prefix boundaries so commas
      inside an answer (e.g. "Yes, because ...") don't break parsing.

    Falls back to per-letter columns (``A``…``E``) if no ``options``
    cell is available.
    """
    if "options" in field_idx:
        cell = row.get(field_idx["options"])
        if cell:
            cell_str = str(cell).strip()
            try:
                parsed = json.loads(cell_str)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return {
                    str(k).strip().upper()[:1]: str(v).strip()
                    for k, v in parsed.items()
                    if str(k).strip().upper()[:1] in _ANSWER_LETTERS
                }
            if isinstance(parsed, list):
                return {
                    letter: str(opt).strip()
                    for letter, opt in zip(_ANSWER_LETTERS, parsed)
                }
            flat = _split_letter_prefixed_options(cell_str)
            if flat:
                return flat

    out: dict[str, str] = {}
    for letter in _ANSWER_LETTERS:
        col = field_idx.get(f"opt_{letter}")
        if col is None:
            continue
        val = row.get(col)
        if val is None:
            continue
        out[letter] = str(val).strip()
    return out


_LETTER_PREFIX_RE = re.compile(r"(?:^|[\s,;])([A-Z])[.)\]:]\s+")


def _split_letter_prefixed_options(text: str) -> dict[str, str]:
    """Parse a single string like ``"A. foo, B. bar, C. baz"``.

    The upstream SIV-Bench TSV uses this concatenated format so the
    whole option list lives inside one TSV cell.  We anchor on letter
    prefixes (``A.``, ``B)``, ``C:`` etc.) and slice the string between
    consecutive anchors so commas inside an answer body are safe.
    Letters in the canonical span ``A``…``L`` (12 options) are kept;
    earlier releases truncated to ``A``…``E`` which silently dropped
    the gold answer for sub-tasks that ship with up to 12 options
    (e.g. Relation Inference).
    """
    matches = list(_LETTER_PREFIX_RE.finditer(text))
    if not matches:
        return {}
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        letter = m.group(1)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip().rstrip(",;").strip()
        if body:
            out[letter] = body
    return {k: v for k, v in out.items() if k in _ANSWER_LETTERS}


# ======================================================================
# GPT-4o parser
# ======================================================================

def _normalise_answer_letter(text: str | None) -> str | None:
    """Extract a single A-E letter from the model's free-form answer."""
    if not text:
        return None
    stripped = text.strip().strip(".").strip()
    if stripped and stripped[0].upper() in _ANSWER_LETTERS:
        candidate = stripped[0].upper()
        if len(stripped) == 1 or not stripped[1].isalpha():
            return candidate
    for letter in _ANSWER_LETTERS:
        for token in (f"{letter}.", f"({letter})", f"[{letter}]",
                      f"Option {letter}", f"option {letter}"):
            if token in stripped:
                return letter
    upper = stripped.upper()
    for letter in _ANSWER_LETTERS:
        if upper.startswith(letter + " ") or upper.startswith(letter + ","):
            return letter
    return None


def parse_siv_bench_sample(
    sample: SIVBenchSample,
    *,
    frames: Sequence[Image.Image] | None = None,
    num_frames: int = 8,
    fps_override: float | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_entities: int = 20,
    max_rounds: int = 6,
    max_side: int = 640,
) -> dict[str, Any]:
    """Run the GPT-4o multi-hop parser on one SIV-Bench question.

    Mirrors ``parse_video_holmes_sample`` so the eval harness can
    iterate both video benchmarks identically.  Returns a dict with the
    schema, normalised answer letter, ground truth, evidence trace, and
    cascade telemetry.
    """
    if frames is None:
        if sample.video_path is None or not sample.video_path.exists():
            raise FileNotFoundError(
                f"Video file missing for {sample.video_id} "
                f"(expected at {sample.video_path}).  See "
                f"install/INSTALL_BENCHMARKS.md §4 (SIV-Bench)."
            )
        frames, fps, video_meta = sample_video_frames(
            sample.video_path, num_frames=num_frames, max_side=max_side,
        )
    else:
        frames = list(frames)
        fps = fps_override or 1.0
        video_meta = {
            "backend": "preloaded",
            "total_frames": len(frames),
            "num_frames": len(frames),
        }

    if fps_override is not None:
        fps = fps_override

    if not frames:
        return {
            "schema": None,
            "answer": None,
            "answer_raw": None,
            "ground_truth": sample.answer,
            "correct": None,
            "tool_trace": [],
            "rounds": 0,
            "num_frames": 0,
            "video_meta": video_meta,
            "model": model,
            "warnings": ["no frames decoded"],
            "sample": sample.to_dict(),
        }

    current_index = len(frames) // 2
    task_id = (
        f"siv_bench.{sample.subtitle}.{sample.video_id}.Q{sample.question_id}"
    )
    routing = classify_question(sample.question, modality="video")
    routing_block = routing.to_prompt_block()
    question_prompt = sample.format_question()
    if routing_block:
        question_prompt = (
            f"{question_prompt}\n\n"
            "Reasoning tools available: count_value, compute_ratio, "
            "compare_values, verify_claim — use them to RECORD any "
            "counts / ratios / comparisons you derive across frames, "
            "and cite the resulting `derivation_id` (d1, d2, …) inside "
            "<derivations> and <answer>.\n"
            f"{routing_block}"
        )

    duration_s = video_meta.get("duration_s") or 0.0
    if duration_s and len(frames) > 0:
        effective_fps = len(frames) / duration_s
    else:
        effective_fps = fps or 1.0
    video_meta["effective_fps"] = round(effective_fps, 6)

    req = GroundingRequest(
        images=frames,
        goal=question_prompt,
        domain="video_qa",
        output_mode="answer",
        task_id=task_id,
        step=0,
        context={
            "fps": effective_fps,
            "native_fps": fps,
            "duration_s": duration_s,
            "sample_timestamps": video_meta.get("sample_timestamps") or [],
            "current_index": current_index,
            "dimension": sample.dimension,
            "subtask": sample.subtask,
            "subtitle": sample.subtitle,
            "options": dict(sample.options),
            "question_classes": routing.classes,
            "required_reasoning_tools": routing.required_tools,
            "derivation_kinds": routing.derivation_kinds,
        },
        max_entities=max_entities,
        max_rounds=max_rounds,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

    primary_size = frames[current_index].size
    result = cascaded_ground(req, image_size=primary_size)

    answer_raw = result.answer
    answer_letter = _normalise_answer_letter(answer_raw)
    gt = sample.answer
    correct: bool | None = None
    if gt and answer_letter:
        correct = answer_letter.upper() == gt.strip().upper()

    return {
        "schema": result.schema,
        "answer": answer_letter,
        "answer_raw": answer_raw,
        "ground_truth": gt,
        "correct": correct,
        "tool_trace": result.tool_trace,
        "rounds": result.rounds,
        "num_frames": len(frames),
        "video_meta": video_meta,
        "model": result.model,
        "raw": result.raw,
        "warnings": result.warnings,
        "validation": result.validation.as_dict() if result.validation else None,
        "head_used": result.head_used,
        "escalation_trace": result.escalation_trace,
        "sample": sample.to_dict(),
    }


# ======================================================================
# Batch helper
# ======================================================================

def parse_siv_bench_batch(
    samples: Iterable[SIVBenchSample],
    *,
    output_jsonl: str | Path | None = None,
    num_frames: int = 8,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_entities: int = 20,
    max_rounds: int = 6,
    temperature: float | None = None,
    max_side: int = 640,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run ``parse_siv_bench_sample`` over a stream of samples."""
    results: list[dict[str, Any]] = []
    fh = None
    if output_jsonl is not None:
        output_path = Path(output_jsonl)
        if output_path.parent and not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(output_path, "a", encoding="utf-8")

    try:
        for i, sample in enumerate(samples, 1):
            try:
                out = parse_siv_bench_sample(
                    sample,
                    num_frames=num_frames,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    max_entities=max_entities,
                    max_rounds=max_rounds,
                    temperature=temperature,
                    max_side=max_side,
                )
            except Exception as exc:
                logger.warning(
                    "SIV-Bench sample %s Q%s failed: %s",
                    sample.video_id, sample.question_id, exc,
                )
                out = {
                    "error": str(exc),
                    "sample": sample.to_dict(),
                }

            results.append(out)
            if fh is not None:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            if progress:
                correct = out.get("correct")
                tag = (
                    "OK" if correct is True
                    else "NO" if correct is False
                    else "??"
                )
                logger.info(
                    "[SIV %s] %d: %s Q%s dim=%s pred=%s gt=%s",
                    tag, i, sample.video_id, sample.question_id,
                    sample.dimension,
                    out.get("answer"),
                    out.get("ground_truth"),
                )
    finally:
        if fh is not None:
            fh.close()

    return results
