"""Per-sample lazy-binding executor wrapper for ``VisualReasoningAdapter``.

Phase-5 Stage 1 follow-up. The canonical bind path
:func:`harness.adapters.visual_reasoning_adapter.bind_visual_reasoning_executor`
takes a single ``image`` and produces one
:class:`~visual_reasoning_wrapper.skill_executor.VisualReasoningExecutor`
for that image. The Stage 1 dispatcher
(:func:`labeling_supplement._phase4_target_dispatch._build_visual_reasoning_target`)
originally left the adapter on its inherited deterministic stub because
it had no way to load the right per-sample image at adapter-construction
time -- the dispatcher fires once per ``--target`` cell, but each
transferred skill is evaluated against a different cold-start sample
that carries its own image.

This module provides:

* :func:`discover_task_to_image` -- walks ``<cold_start_root>/<run>/<sub_corpus>/``
  to cross-reference canonical ``sample_*.json`` files with their
  ``frames/sample_NNN/frame_00.png`` siblings, returning a
  ``{task_id: image_path}`` map.

* :class:`TaskAwareVisualReasoningExecutor` -- a ``HopExecutor``-compatible
  wrapper that takes that map plus a fallback to the deterministic stub.
  On each hop, it inspects ``ctx.state.task`` to look up the right image,
  lazily constructs (and caches) a real
  :class:`~visual_reasoning_wrapper.skill_executor.VisualReasoningExecutor`
  for that image, and delegates. Falls back to the deterministic stub when
  the task_id is absent from the mapping or the image file is missing on
  disk -- this keeps the wire exercisable in degraded conditions, matching
  the prior behaviour of leaving everything on the stub.

The image cache is keyed by ``image_path`` so a hot loop of hops over the
same image hits the cache; building the executor for a new image is the
expensive step (it materialises the visual + reasoning tool registries
under :mod:`visual_reasoning_wrapper.tools_visual` /
:mod:`visual_reasoning_wrapper.tools_reasoning`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("harness.vr_per_sample_executor")

__all__ = [
    "TaskAwareVisualReasoningExecutor",
    "discover_task_to_image",
]


def discover_task_to_image(
    cold_start_root: Path,
    *,
    sub_corpus: str,
) -> Dict[str, Path]:
    """Build a ``{task_id: image_path}`` map for one cold-start sub-corpus.

    Walks every timestamped run directory under ``cold_start_root``
    (newest first), and for each ``<run>/<sub_corpus>/sample_NNN.json``
    that has a sibling ``<run>/<sub_corpus>/frames/sample_NNN/frame_00.png``,
    records ``{sample.task_id: frame_path}``. The newest run wins on ties
    so a re-run of the same sample picks up the freshest frame.

    Returns an empty dict if ``cold_start_root`` does not exist or no
    matching frames are found -- callers should treat that as "stay on
    the deterministic stub" (see :class:`TaskAwareVisualReasoningExecutor`'s
    fallback path).

    The on-disk layout this expects is what
    ``Cold-start-out-visual-reasoning/`` ships today:

    .. code-block::

        Cold-start-out-visual-reasoning/
        +-- 2026-04-29_00-47-53/
        |   +-- visual_toolbench/
        |   |   +-- sample_000.json    <-- carries {task_id, sample_id}
        |   |   +-- frames/
        |   |       +-- sample_000/
        |   |           +-- frame_00.png
        |   +-- tir_bench/
        |       +-- ...
        +-- 2026-04-29_04-20-45/
        +-- visual_toolbench/             <-- canonical (no frames sibling)
        +-- tir_bench/
    """
    cold_start_root = Path(cold_start_root)
    if not cold_start_root.exists():
        return {}

    task_to_image: Dict[str, Path] = {}
    # Run dirs are timestamped 'YYYY-MM-DD_HH-MM-SS'; sort newest first
    # so the most recent frame for any given task_id wins.
    run_dirs = sorted(
        [
            d for d in cold_start_root.iterdir()
            if d.is_dir() and d.name.startswith("20")
        ],
        reverse=True,
    )
    for run in run_dirs:
        corpus_dir = run / sub_corpus
        frames_dir = corpus_dir / "frames"
        if not corpus_dir.is_dir() or not frames_dir.is_dir():
            continue
        for sample_json in corpus_dir.glob("sample_*.json"):
            try:
                payload = json.loads(sample_json.read_text())
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "skipping unreadable sample %s: %r",
                    sample_json, exc,
                )
                continue
            tid = payload.get("task_id") or ""
            if not tid:
                continue
            frame = frames_dir / sample_json.stem / "frame_00.png"
            if frame.exists() and tid not in task_to_image:
                task_to_image[tid] = frame
    logger.info(
        "discovered %d task_id->image mapping(s) for sub_corpus=%s under %s",
        len(task_to_image), sub_corpus, cold_start_root,
    )
    return task_to_image


class TaskAwareVisualReasoningExecutor:
    """``HopExecutor`` wrapper with per-sample lazy executor binding.

    Suitable for the Stage 1 (image-VR) dispatcher where each transferred
    skill is evaluated against a different cold-start sample, and each
    sample carries its own image. Caches one real
    :class:`~visual_reasoning_wrapper.skill_executor.VisualReasoningExecutor`
    per image path so a hot loop of hops on the same image hits the cache.

    Falls back to the deterministic stub from
    :func:`harness.adapters._stub_base.make_deterministic_executor` when:

    * ``ctx.state.task`` is empty / not in ``task_to_image``
    * the mapped image path doesn't exist on disk
    * ``PIL.Image.open(...).load()`` raises (corrupt / unsupported file)
    * ``VisualReasoningExecutor.from_image(...)`` raises

    The fallback is intentionally permissive: the goal of this wrapper is
    to *upgrade* hops where a real image is available without breaking
    the chain on hops where it isn't. The dispatcher logs the discovery
    summary so degraded coverage is surfaced.
    """

    def __init__(
        self,
        task_to_image: Dict[str, Path],
        *,
        prefer_gdino: bool = False,
        confidence: float = 0.8,
    ) -> None:
        self._task_to_image = dict(task_to_image)
        self._prefer_gdino = prefer_gdino
        self._confidence = confidence
        self._executor_cache: Dict[Path, Any] = {}
        # Lazily constructed deterministic stub for fallbacks (avoids
        # importing the stub module at construction time).
        self._stub: Optional[Any] = None

    def task_count(self) -> int:
        """Number of distinct ``task_id``s with a mapped image."""
        return len(self._task_to_image)

    def _stub_executor(self) -> Any:
        if self._stub is None:
            from harness.adapters._stub_base import make_deterministic_executor
            self._stub = make_deterministic_executor("visual_reasoning")
        return self._stub

    def _executor_for(self, image_path: Path) -> Any:
        """Get or build the cached executor for ``image_path``.

        On any failure (decode error, executor build error) falls back to
        the deterministic stub and caches THAT against ``image_path`` so
        we don't retry the failing path on every subsequent hop.
        """
        if image_path in self._executor_cache:
            return self._executor_cache[image_path]
        from PIL import Image
        from visual_reasoning_wrapper.skill_executor import (
            make_visual_reasoning_executor,
        )
        try:
            img = Image.open(image_path)
            img.load()  # eager decode so any failure surfaces here
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PIL.Image.open(%s) failed (%r); using stub for this task",
                image_path, exc,
            )
            self._executor_cache[image_path] = self._stub_executor()
            return self._executor_cache[image_path]
        try:
            ex = make_visual_reasoning_executor(
                img,
                prefer_gdino=self._prefer_gdino,
                confidence=self._confidence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VisualReasoningExecutor.from_image(%s) failed (%r); "
                "using stub for this task",
                image_path, exc,
            )
            ex = self._stub_executor()
        self._executor_cache[image_path] = ex
        return ex

    def __call__(self, action_type: str, payload: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
        task = getattr(getattr(ctx, "state", None), "task", None) or ""
        image_path = self._task_to_image.get(task)
        if image_path is None or not image_path.exists():
            return self._stub_executor()(action_type, payload, ctx)
        return self._executor_for(image_path)(action_type, payload, ctx)
