"""Per-sample lazy-binding executor wrapper for ``VideoAdapter``.

Phase-5 Stage 2 follow-up. The cross-domain video transfer cell is
authored once per ``--target`` (``video_holmes`` or ``siv_bench``) but
each transferred skill is evaluated against a different cold-start
sample that carries its own video clip + sampled-frame metadata under
``video_meta``. This wrapper bridges the gap by:

* :func:`discover_task_to_video_meta` -- walks
  ``<cold_start_root>/<sub_corpus>/sample_*.json`` once at dispatcher
  build-time and returns ``{task_id: video_meta_dict}``.

* :class:`TaskAwareVideoReasoningExecutor` -- a ``HopExecutor``-shaped
  callable that, on each hop, looks up ``ctx.state.task`` against that
  map. When the task has a real ``video_meta`` and the underlying
  ``video_path`` exists on disk:

  - **InnerAction verbs** (``GROUND`` / ``RETRIEVE`` / ``CHECK`` /
    ``VERIFY`` / ``COMMIT`` / ``EXECUTE``) are routed to a real
    :class:`~visual_reasoning_wrapper.video_skill_executor.VideoReasoningExecutor`
    that decodes the cold-start sampled frames via
    :func:`visual_reasoning_wrapper.benchmarks.video_holmes.sample_video_frames`
    (uniform sampling over ``total_frames`` -- deterministic, so we
    re-decode the SAME indices the cold-start labeller saw).

  - **Legacy / video-domain verbs** (``SAMPLE_FRAME``, ``INSPECT_FRAME``,
    ``EMIT_ANSWER``, ``CHECK_RELATION``, ``OCR``, ...) and any unknown
    op are routed to a per-task
    :func:`~harness.video_executor.make_video_executor` deterministic
    stub that has the same ``video_meta`` wired in. This keeps the
    chain firing for cold-start protocols that emit the legacy verbs
    while still upgrading the ``InnerAction`` hops to real VLM tools.

Falls back to a bare deterministic stub when (a) ``ctx.state.task`` is
empty / not in the mapping, (b) the mapped ``video_path`` is missing on
disk, or (c) frame decode / executor construction raises -- mirroring
:class:`harness._vr_per_sample_executor.TaskAwareVisualReasoningExecutor`'s
permissive fallback policy.

The real-executor cache is keyed by ``video_path`` so a hot loop of
hops over the same clip hits the cache; building the executor for a
new clip is the expensive step (frame decode + tool registry
construction).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("harness.video_per_sample_executor")

__all__ = [
    "INNER_ACTION_VERBS",
    "TaskAwareVideoReasoningExecutor",
    "discover_task_to_video_meta",
]


#: The InnerAction verbs the real ``VideoReasoningExecutor`` knows how
#: to dispatch onto concrete tools. Anything outside this set falls
#: through to the per-task deterministic stub so cold-start protocols
#: that use the video-domain verb set
#: (``SAMPLE_FRAME`` / ``EMIT_ANSWER`` / ...) still emit typed evidence.
INNER_ACTION_VERBS: frozenset[str] = frozenset({
    "GROUND", "RETRIEVE", "CHECK", "VERIFY", "COMMIT", "EXECUTE",
})


def discover_task_to_video_meta(
    cold_start_root: Path,
    *,
    sub_corpus: str,
) -> Dict[str, Dict[str, Any]]:
    """Build a ``{task_id: video_meta_dict}`` map for one video sub-corpus.

    Walks ``<cold_start_root>/<sub_corpus>/sample_*.json`` and records
    the first ``video_meta`` payload per task_id. Returns an empty dict
    when the root or sub-corpus directory is missing -- callers should
    treat that as "stay on the deterministic stub" (see
    :class:`TaskAwareVideoReasoningExecutor`'s fallback path).

    The on-disk layout this expects is what
    ``Cold-start-out-visual-reasoning-video/`` ships today:

    .. code-block::

        Cold-start-out-visual-reasoning-video/
        +-- video_holmes/
        |   +-- sample_000.json    <-- carries {task_id, video_meta:{video_path, indices, sample_timestamps, ...}}
        |   +-- sample_001.json
        |   +-- ...
        +-- siv_bench/
            +-- ...

    Unlike the image case (``cold_start_root/<run>/<sub_corpus>/``) the
    video tree is flat -- one corpus dir per sub_corpus, sample files
    directly inside, no run-id timestamp. We sort by name so re-runs are
    deterministic.
    """
    cold_start_root = Path(cold_start_root)
    if not cold_start_root.exists():
        return {}
    corpus_dir = cold_start_root / sub_corpus
    if not corpus_dir.is_dir():
        return {}

    task_to_meta: Dict[str, Dict[str, Any]] = {}
    for sample_json in sorted(corpus_dir.glob("sample_*.json")):
        try:
            payload = json.loads(sample_json.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "skipping unreadable sample %s: %r", sample_json, exc,
            )
            continue
        tid = payload.get("task_id") or ""
        if not tid:
            continue
        meta = payload.get("video_meta") or {}
        if not isinstance(meta, dict) or not meta.get("video_path"):
            continue
        if tid not in task_to_meta:
            task_to_meta[tid] = dict(meta)
    logger.info(
        "discovered %d task_id->video_meta mapping(s) for sub_corpus=%s under %s",
        len(task_to_meta), sub_corpus, cold_start_root,
    )
    return task_to_meta


class TaskAwareVideoReasoningExecutor:
    """``HopExecutor`` wrapper with per-sample lazy executor binding.

    Suitable for the Stage 2 (video-VR) dispatcher where each
    transferred skill is evaluated against a different cold-start
    sample, and each sample carries its own clip. Caches one real
    :class:`~visual_reasoning_wrapper.video_skill_executor.VideoReasoningExecutor`
    per ``video_path`` so a hot loop of hops on the same clip hits the
    cache.

    Routing rules per hop:

    1. ``ctx.state.task`` not in map -> bare deterministic stub
       (``make_video_executor(video_meta=None)``).
    2. ``video_path`` missing on disk -> bare deterministic stub.
    3. ``action_type`` outside :data:`INNER_ACTION_VERBS` -> per-task
       deterministic stub (``make_video_executor(video_meta=meta)``)
       so legacy / video-domain verbs still emit typed evidence carrying
       the right ``video_path`` in their payload.
    4. Real ``VideoReasoningExecutor`` build failure (frame decode
       error, registry construction error) -> per-task stub for THIS
       video (cached against ``video_path``).
    5. Otherwise -> real ``VideoReasoningExecutor`` for this clip.
    """

    def __init__(
        self,
        task_to_video_meta: Dict[str, Dict[str, Any]],
        *,
        num_frames: int = 8,
        max_side: int = 640,
        prefer_gdino: bool = False,
        confidence: float = 0.8,
    ) -> None:
        self._task_to_meta = dict(task_to_video_meta)
        self._num_frames = int(num_frames)
        self._max_side = int(max_side)
        self._prefer_gdino = bool(prefer_gdino)
        self._confidence = float(confidence)
        # Caches keyed by `video_path`. Real-executor cache holds the
        # successfully-built `VideoReasoningExecutor` (or None on a
        # decode/build failure -- in which case the stub-cache entry
        # for that path is consulted instead).
        self._real_cache: Dict[str, Optional[Any]] = {}
        self._stub_cache: Dict[str, Any] = {}
        self._bare_stub: Optional[Any] = None

    def task_count(self) -> int:
        """Number of distinct ``task_id``s with a mapped ``video_meta``."""
        return len(self._task_to_meta)

    # ------------------------------------------------------------------
    # Stub helpers
    # ------------------------------------------------------------------

    def _bare_stub_executor(self) -> Any:
        if self._bare_stub is None:
            from harness.video_executor import make_video_executor
            self._bare_stub, _holder = make_video_executor(video_meta=None)
        return self._bare_stub

    def _stub_for(self, video_meta: Dict[str, Any]) -> Any:
        """Per-task stub that carries ``video_meta`` in evidence payloads."""
        key = str(video_meta.get("video_path") or "")
        if key in self._stub_cache:
            return self._stub_cache[key]
        from harness.video_executor import make_video_executor
        executor, _holder = make_video_executor(video_meta=video_meta)
        self._stub_cache[key] = executor
        return executor

    # ------------------------------------------------------------------
    # Real executor construction (cached per video_path)
    # ------------------------------------------------------------------

    def _real_for(self, video_meta: Dict[str, Any]) -> Optional[Any]:
        key = str(video_meta.get("video_path") or "")
        if key in self._real_cache:
            return self._real_cache[key]
        # Lazy import so the stub-only path doesn't pull cv2 / decord.
        try:
            from visual_reasoning_wrapper.benchmarks.video_holmes import (
                sample_video_frames,
            )
            from visual_reasoning_wrapper.video_skill_executor import (
                VideoReasoningExecutor,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "video imports failed (%r); falling back to per-task stub",
                exc,
            )
            self._real_cache[key] = None
            return None

        num_frames = int(video_meta.get("num_frames") or self._num_frames)
        try:
            frames, native_fps, decoded_meta = sample_video_frames(
                video_meta["video_path"],
                num_frames=num_frames,
                max_side=self._max_side,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sample_video_frames(%s) failed (%r); using stub for this task",
                video_meta.get("video_path"), exc,
            )
            self._real_cache[key] = None
            return None
        if not frames:
            logger.warning(
                "sample_video_frames(%s) returned 0 frames; using stub",
                video_meta.get("video_path"),
            )
            self._real_cache[key] = None
            return None

        # Prefer the cold-start sample's pre-computed sample_timestamps
        # (the actual seconds in the original clip the labeller saw)
        # over the freshly-decoded ones, so per-frame timestamp probes
        # match what the protocol-lift was trained against.
        sample_timestamps = (
            list(video_meta.get("sample_timestamps") or [])
            or list(decoded_meta.get("sample_timestamps") or [])
            or []
        )
        # Effective fps: frames-decoded / clip-duration so
        # `sample_frames(start_sec, end_sec)` in the registry lines up
        # with the actual video timeline. Mirrors the cold-start
        # labeller's `effective_fps` math in benchmarks/video_holmes.py.
        duration_s = float(
            video_meta.get("duration_s")
            or decoded_meta.get("duration_s")
            or 0.0
        )
        if duration_s > 0:
            effective_fps = len(frames) / duration_s
        else:
            effective_fps = float(native_fps or 1.0)

        try:
            executor = VideoReasoningExecutor.from_frames(
                frames,
                fps=effective_fps,
                sample_timestamps=sample_timestamps,
                prefer_gdino=self._prefer_gdino,
                confidence=self._confidence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VideoReasoningExecutor.from_frames(%s) failed (%r); "
                "using stub for this task",
                video_meta.get("video_path"), exc,
            )
            self._real_cache[key] = None
            return None
        self._real_cache[key] = executor
        return executor

    # ------------------------------------------------------------------
    # HopExecutor protocol
    # ------------------------------------------------------------------

    def __call__(
        self,
        action_type: str,
        payload: Dict[str, Any],
        ctx: Any,
    ) -> Dict[str, Any]:
        task = getattr(getattr(ctx, "state", None), "task", None) or ""
        video_meta = self._task_to_meta.get(task)
        if not video_meta:
            return self._bare_stub_executor()(action_type, payload, ctx)
        path = video_meta.get("video_path")
        if not path or not Path(path).exists():
            return self._bare_stub_executor()(action_type, payload, ctx)

        op = (action_type or "").upper()
        if op not in INNER_ACTION_VERBS:
            # Legacy / video-domain verb -- per-task stub keeps the
            # video_path in evidence and the answer in state.facts.
            return self._stub_for(video_meta)(action_type, payload, ctx)

        real = self._real_for(video_meta)
        if real is None:
            return self._stub_for(video_meta)(action_type, payload, ctx)
        return real(action_type, payload, ctx)
