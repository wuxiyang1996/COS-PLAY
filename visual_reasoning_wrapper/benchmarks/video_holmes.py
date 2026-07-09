"""Video-Holmes loader + GPT-4o parser.

Video-Holmes is a 1 837-question multi-hop video-reasoning benchmark
built on 503 cropped suspense-film clips. Each question is a 6-way
multiple choice whose answer letter (``A``–``F``) maps into a rich
``Options`` dict. The benchmark's difficulty comes from requiring the
model to connect several clues scattered across the clip, which makes
it a natural fit for our multi-hop evidence-chain schema
(``<evidence>`` + ``<answer>``).

Disk layout expected at ``default_video_holmes_root()``:

```
<root>/Benchmark/
    videos/videos_cropped/<video_id>.mp4
    test_Video-Holmes.json
    train_Video-Holmes.json
    annotations/<video_id>.json            # per-clip annotations
    annotation_training/<video_id>.json
```

Usage::

    from visual_reasoning_wrapper.benchmarks.video_holmes import (
        iter_video_holmes_samples, parse_video_holmes_sample,
    )

    for sample in iter_video_holmes_samples(split="test", limit=3):
        out = parse_video_holmes_sample(
            sample, num_frames=8, model="gpt-4o", api_key=KEY,
        )
        print(sample.question_type, out["answer"], "gt:", sample.answer)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from vlm_wrapper.ground import GroundingRequest, cascaded_ground
from ..question_router import classify_question

logger = logging.getLogger(__name__)

_QUESTION_FILES: dict[str, str] = {
    "test": "test_Video-Holmes.json",
    "train": "train_Video-Holmes.json",
}

_ANSWER_LETTERS = ("A", "B", "C", "D", "E", "F")


@dataclass
class VideoHolmesSample:
    """One Video-Holmes question + the matching clip on disk.

    Field names are normalised (no spaces) so downstream code can use
    attribute access.
    """

    video_id: str
    question_id: int
    question_type: str
    question: str
    options: dict[str, str]
    answer: str | None = None
    explanation: str | None = None
    video_path: Path | None = None
    annotation_path: Path | None = None
    split: str = "test"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question": self.question,
            "options": dict(self.options),
            "answer": self.answer,
            "explanation": self.explanation,
            "video_path": str(self.video_path) if self.video_path else None,
            "annotation_path": (
                str(self.annotation_path) if self.annotation_path else None
            ),
            "split": self.split,
        }

    def format_question(self) -> str:
        """Render the MCQ prompt the way Video-Holmes evaluates it."""
        lines = [self.question.strip(), "", "Options:"]
        for letter in _ANSWER_LETTERS:
            opt = self.options.get(letter)
            if opt is None:
                continue
            lines.append(f"{letter}. {opt}")
        lines.append("")
        lines.append(
            "Procedure (follow strictly):\n"
            "1. FIRST hop must be temporal — call `find_moment` or "
            "`detect_scene_changes` to localise the relevant clip "
            "before reasoning about objects.  Skipping this step makes "
            "<entities> ungrounded.\n"
            "2. Then call `sample_frames` and `detect_objects_at_frame` "
            "(or `read_text_in_frame`) on the localised window.\n"
            "3. In <entities>, list the people / objects / events you "
            "detected, each with `pos=x,y,w,h` (bbox in the sampled "
            "frame) or `null`, and an `ontology=` type.  In <targets>, "
            "`target=` is the entity ID of the ANSWER referent; "
            "`candidate_set=[e1,e2,...]` is a list of ENTITY IDs you "
            "considered (NOT the MCQ option letters A–F).\n"
            "4. Each <evidence> hop must declare `abstract_op=` "
            "(GROUND / CHECK / RETRIEVE / CONCLUDE) and `tool=` of the "
            "ACTUAL tool you called — do NOT invent hops.  Include "
            "`frame=<idx>` and `timestamp=<s>` for every temporal hop.\n"
            "5. In <state_flags>, set `scene_type=video_segment`, "
            "`progress=null`, `phase=null`, `dialog_open=false`, "
            "`input_pending=false`.\n"
            "6. Produce the full <state>…</state> schema defined in the "
            "system prompt.  Inside the <answer> block, put a single "
            "letter (A-F) on the line beginning with 'answer=' — for "
            "example 'answer=C'."
        )
        return "\n".join(lines)


# ======================================================================
# Disk layout helpers
# ======================================================================

def default_video_holmes_root(
    workspace_root: str | Path | None = None,
) -> Path:
    """Return the canonical Video-Holmes root on this workspace."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parents[2]
    return Path(workspace_root) / "data" / "Video-Holmes"


def _benchmark_dir(root: Path) -> Path:
    return root / "Benchmark"


def _video_path(root: Path, video_id: str) -> Path:
    return _benchmark_dir(root) / "videos" / "videos_cropped" / f"{video_id}.mp4"


def _annotation_path(root: Path, video_id: str, split: str) -> Path:
    subdir = "annotations" if split == "test" else "annotation_training"
    return _benchmark_dir(root) / subdir / f"{video_id}.json"


# ======================================================================
# Loaders
# ======================================================================

def load_video_holmes_questions(
    split: str = "test",
    *,
    video_holmes_root: str | Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load the raw question list for a split."""
    root = (
        Path(video_holmes_root)
        if video_holmes_root else default_video_holmes_root()
    )
    fname = _QUESTION_FILES.get(split)
    if fname is None:
        raise ValueError(
            f"split must be one of {list(_QUESTION_FILES)}, got {split!r}"
        )
    qpath = _benchmark_dir(root) / fname
    if not qpath.exists():
        raise FileNotFoundError(
            f"Video-Holmes questions not found at {qpath}. "
            f"Download per install/INSTALL_BENCHMARKS.md §5."
        )
    with qpath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def iter_video_holmes_samples(
    split: str = "test",
    *,
    video_holmes_root: str | Path | None = None,
    limit: int | None = None,
    video_ids: Iterable[str] | None = None,
    question_types: Iterable[str] | None = None,
) -> Iterator[VideoHolmesSample]:
    """Yield ``VideoHolmesSample`` objects for a split.

    Parameters
    ----------
    split : str
        ``"test"`` or ``"train"``.
    limit : int, optional
        Stop after this many samples.
    video_ids : iterable of str, optional
        Restrict to a whitelist of video IDs.
    question_types : iterable of str, optional
        Restrict to question types such as ``SR``, ``IMC``, ``TCI``,
        ``TA``, ``MHR``, ``PAR``, ``CTI`` (see Video-Holmes README).
    """
    root = (
        Path(video_holmes_root)
        if video_holmes_root else default_video_holmes_root()
    )

    vid_set = set(video_ids) if video_ids else None
    type_set = set(question_types) if question_types else None

    questions = load_video_holmes_questions(
        split, video_holmes_root=root, limit=None,
    )

    count = 0
    for q in questions:
        video_id = q.get("video ID") or q.get("video_id") or ""
        qtype = q.get("Question Type") or q.get("question_type") or ""
        if vid_set is not None and video_id not in vid_set:
            continue
        if type_set is not None and qtype not in type_set:
            continue

        vpath = _video_path(root, video_id)
        apath = _annotation_path(root, video_id, split)

        sample = VideoHolmesSample(
            video_id=video_id,
            question_id=int(q.get("Question ID", q.get("question_id", 0))),
            question_type=qtype,
            question=q.get("Question", q.get("question", "")),
            options=dict(q.get("Options", q.get("options", {}))),
            answer=q.get("Answer", q.get("answer")),
            explanation=q.get("Explanation", q.get("explanation")),
            video_path=vpath if vpath.exists() else None,
            annotation_path=apath if apath.exists() else None,
            split=split,
        )
        yield sample
        count += 1
        if limit is not None and count >= limit:
            return


# ======================================================================
# Frame sampling — runs whichever decoder is installed
# ======================================================================

def _open_video(path: str | Path) -> tuple[Any, str]:
    """Return ``(reader, backend)`` where backend is 'decord' or 'cv2'."""
    path = str(path)
    try:
        import decord  # type: ignore
        reader = decord.VideoReader(path)
        return reader, "decord"
    except Exception:
        pass

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "No video decoder found. Install one of: "
            "`pip install decord` (preferred) or "
            "`pip install opencv-python`."
        ) from exc

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    return cap, "cv2"


def _sample_indices(total_frames: int, num_frames: int) -> list[int]:
    """Uniformly pick ``num_frames`` indices over [0, total_frames)."""
    if total_frames <= 0 or num_frames <= 0:
        return []
    if num_frames >= total_frames:
        return list(range(total_frames))
    # Use the midpoints of ``num_frames`` equal-sized buckets so we
    # avoid picking the very first / very last frame (often a slate).
    return [
        int((i + 0.5) * total_frames / num_frames)
        for i in range(num_frames)
    ]


def sample_video_frames(
    video_path: str | Path,
    num_frames: int = 8,
    *,
    max_side: int = 640,
) -> tuple[list[Image.Image], float, dict[str, Any]]:
    """Uniformly sample ``num_frames`` PIL frames from a video.

    Returns ``(frames, fps, meta)`` where ``meta`` carries
    ``total_frames``, ``duration``, ``backend``, ``indices``, and
    ``size`` so callers can surface the sampling info to the VLM.

    Frames are down-scaled so that the longest edge is ``max_side``
    pixels to keep API costs reasonable. Pass ``max_side=0`` to disable
    resizing.
    """
    reader, backend = _open_video(video_path)
    frames: list[Image.Image] = []
    indices: list[int] = []

    try:
        if backend == "decord":
            total = len(reader)
            fps = float(reader.get_avg_fps() or 0.0)
            indices = _sample_indices(total, num_frames)
            if indices:
                batch = reader.get_batch(indices).asnumpy()
                for arr in batch:
                    frames.append(_to_pil(arr, max_side))
        else:
            import cv2  # type: ignore
            total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(reader.get(cv2.CAP_PROP_FPS) or 0.0)
            indices = _sample_indices(total, num_frames)
            for idx in indices:
                reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = reader.read()
                if not ok or frame is None:
                    continue
                # cv2 returns BGR; PIL expects RGB
                frame = frame[:, :, ::-1]
                frames.append(_to_pil(frame, max_side))
    finally:
        if backend == "cv2":
            reader.release()

    duration = (total / fps) if (fps and fps > 0) else None
    # Real per-sample timestamps in the *original* video so downstream
    # tools can expose wallclock seconds instead of "sampled-frame
    # index / native fps" (which would pretend the whole video is
    # 0.3 seconds long for an 8-sample / 250-s clip).
    sample_timestamps: list[float] = (
        [round(idx / fps, 3) for idx in indices]
        if (indices and fps and fps > 0)
        else []
    )
    meta = {
        "backend": backend,
        "total_frames": total,
        "duration_s": duration,
        "native_fps": fps or None,
        "indices": indices,
        "sample_timestamps": sample_timestamps,
        "num_frames": len(frames),
        "size": frames[0].size if frames else None,
    }
    return frames, fps, meta


def _to_pil(arr: np.ndarray, max_side: int) -> Image.Image:
    img = Image.fromarray(arr)
    if max_side and max_side > 0:
        w, h = img.size
        long = max(w, h)
        if long > max_side:
            scale = max_side / long
            img = img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS,
            )
    return img


# ======================================================================
# GPT-4o parser
# ======================================================================

def _normalise_answer_letter(text: str | None) -> str | None:
    """Extract a single A–F letter from the model's free-form answer."""
    if not text:
        return None
    stripped = text.strip().strip(".").strip()
    if len(stripped) >= 1 and stripped[0].upper() in _ANSWER_LETTERS:
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


def parse_video_holmes_sample(
    sample: VideoHolmesSample,
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
    """Run the GPT-4o multi-hop parser on one Video-Holmes question.

    The sample's clip is sampled into ``num_frames`` frames (uniform
    over the timeline) and fed to ``ground`` with ``domain="video_qa"``.
    The VLM sees the middle frame directly and can navigate the rest of
    the clip through the video-visual tool registry (temporal
    navigation, per-frame detection, cross-frame tracking).

    Parameters
    ----------
    sample : VideoHolmesSample
        One row from ``iter_video_holmes_samples``.
    frames : sequence of PIL.Image, optional
        Pre-sampled frames. Bypasses the internal video decoder, useful
        for tests.
    num_frames : int
        Number of frames to sample when ``frames`` is ``None``.
    fps_override : float, optional
        Override the real clip FPS (not usually needed).
    model, api_key, base_url, temperature :
        Pass-through overrides for OpenAI.
    max_entities, max_rounds : int
        Schema / loop caps.
    max_side : int
        Longest-edge resize for sampled frames (0 disables).

    Returns
    -------
    dict with keys:
        ``schema``             – raw ``<state>…</state>`` text
        ``answer_raw``         – the ``answer=`` value produced by GPT-4o
        ``answer``             – normalised letter A–F (or ``None``)
        ``ground_truth``       – ``sample.answer``
        ``correct``            – ``None`` / True / False
        ``tool_trace``         – list of tool calls
        ``rounds``             – number of loop rounds
        ``num_frames``         – frame count actually sent to the model
        ``video_meta``         – dict from ``sample_video_frames``
        ``model``              – model string used
        ``sample``             – ``sample.to_dict()``
    """
    if frames is None:
        if sample.video_path is None or not sample.video_path.exists():
            raise FileNotFoundError(
                f"Video file missing for {sample.video_id} "
                f"(expected at {sample.video_path}). Run "
                f"install/INSTALL_BENCHMARKS.md §5.4 to extract videos.zip."
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
    task_id = f"video_holmes.{sample.split}.{sample.video_id}.Q{sample.question_id}"
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

    # Derive an *effective* fps for the downsampled frame list so the
    # video tools (sample_frames / detect_scene_changes / get_video_info)
    # surface wallclock timestamps instead of native-fps timestamps.
    # Without this, an 8-frame sample of a 250-second clip pretends to
    # be a 1/3-second micro-clip and the LLM asks for more frames inside
    # a tiny 0.3-s window, misses the rest of the movie, and hallucinates
    # context from essentially one frame.
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
            "question_type": sample.question_type,
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

def parse_video_holmes_batch(
    samples: Iterable[VideoHolmesSample],
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
    """Run ``parse_video_holmes_sample`` over a stream of samples."""
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
                out = parse_video_holmes_sample(
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
                    "Video-Holmes sample %s Q%s failed: %s",
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
                    "[VH %s] %d: %s Q%s type=%s pred=%s gt=%s",
                    tag, i, sample.video_id, sample.question_id,
                    sample.question_type,
                    out.get("answer"),
                    out.get("ground_truth"),
                )
    finally:
        if fh is not None:
            fh.close()

    return results
