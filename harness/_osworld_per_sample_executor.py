"""Per-sample lazy-binding executor wrapper for ``OsworldAdapter``.

Phase-5 §12.1 item 3 (Tier 1 OSWorld real-env binding). Mirrors the
Stage 1 image-VR pattern shipped via
:mod:`harness._vr_per_sample_executor` and the Stage 2 video pattern
shipped via :mod:`harness._video_per_sample_executor`, applied to the
OSWorld desktop transfer cell.

The cross-domain video / image-VR cells transferred against
*pre-recorded* media; the OSWorld cell transfers against *live*
``happysixd/osworld-docker`` containers reached via HTTP (see
:mod:`harness._executor_helpers.osworld_client`). A pool of N
containers (typically 13 in this workspace) is hash-pinned per
``task_id`` so a hot loop of hops on the same task keeps hitting the
same container's desktop state.

Routing rules per hop (mirrors the video executor's verb routing):

1. ``ctx.state.task`` not in map  ->  bare deterministic stub
   (``make_osworld_executor()``).
2. Container fleet empty / all unreachable  ->  bare deterministic
   stub. A previous ``discover_running_containers`` failure is
   sticky for this executor lifetime.
3. ``action_type`` outside :data:`INNER_ACTION_VERBS` and outside
   :data:`PRIMITIVE_DESKTOP_OPS`  ->  per-task deterministic stub
   so legacy / cross-domain protocol verbs (``INSPECT``, ``RECALL_*``,
   etc.) still emit typed evidence.
4. ``action_type`` in :data:`PRIMITIVE_DESKTOP_OPS` (``CLICK``,
   ``TYPE``, ``HOTKEY``, ...)  ->  translated to a pyautogui code
   string and POSTed to the pinned container via
   :meth:`OsworldClient.run_pyautogui`. Returns evidence carrying
   the container name, the host port, and the pyautogui status.
5. ``action_type`` in :data:`INNER_ACTION_VERBS` (``GROUND``,
   ``RETRIEVE``, ``CHECK``, ``VERIFY``, ``COMMIT``, ``EXECUTE``)
   ->  current screenshot is fetched from the pinned container,
   a :class:`VisualReasoningExecutor` is built / cached against
   that screenshot, and the hop is dispatched to it.

The screenshot cache is keyed by ``(container_port, screenshot_hash)``
so consecutive visual hops on the same desktop state reuse one
``VisualReasoningExecutor`` (avoiding redundant OmniParser-v2 /
Florence-2 dispatch on the same image), while a fresh screenshot
after a desktop-mutating verb invalidates the cache.

The executor falls back to the deterministic stub on any failure
(container unreachable, screenshot decode error, executor build
failure, pyautogui-code execution error) -- mirroring the permissive
fallback policy of
:class:`harness._video_per_sample_executor.TaskAwareVideoReasoningExecutor`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("harness.osworld_per_sample_executor")

__all__ = [
    "INNER_ACTION_VERBS",
    "PRIMITIVE_DESKTOP_OPS",
    "TaskAwareOsworldExecutor",
    "discover_task_to_osworld_meta",
]


#: InnerAction verbs the real :class:`VisualReasoningExecutor` knows
#: how to dispatch. Anything outside this set falls through to either
#: the primitive-desktop pyautogui path or the per-task deterministic
#: stub.
INNER_ACTION_VERBS: frozenset[str] = frozenset({
    "GROUND", "RETRIEVE", "CHECK", "VERIFY", "COMMIT", "EXECUTE",
})

#: Desktop-primitive verbs that map onto pyautogui code. ``EXECUTE``
#: is the canonical InnerAction verb for "perform the underlying
#: action"; the executor picks pyautogui-code dispatch for ``EXECUTE``
#: when the payload carries an explicit ``code`` field.
PRIMITIVE_DESKTOP_OPS: frozenset[str] = frozenset({
    "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK",
    "TYPE", "KEY_PRESS", "HOTKEY",
    "SCROLL", "DRAG", "MOVE_MOUSE",
    "WAIT", "DONE", "FINISH",
})


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_task_to_osworld_meta(
    cold_start_root: Path,
    *,
    domain_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Walk ``Cold-start-out-osworld/<run>/<domain>/<task_uuid>/episode_*.json``
    and return ``{task_id: meta_dict}``.

    Each meta dict carries::

        {
            "domain": "vlc",
            "task_id": "215dfd39-...",
            "episode_path": "Cold-start-out-osworld/.../episode_000.json",
            "frames_dir": "Cold-start-out-osworld/.../frames/ep_000",
            "first_action": "pyautogui.click(947, 374)",
            "goal": "OSWorld task (vlc): Can you disable the cone icon...",
        }

    The ``episode_path`` and ``frames_dir`` give callers a fallback
    path to render evidence even when the live container fleet is
    unreachable (the ``frames_dir/step_NNN.png`` images are the
    same screenshots the cold-start labeller saw).

    Walks the most recent run for each ``(domain, task_id)`` pair so
    re-running the cold-start labeller without changing the corpus
    just refreshes the records rather than duplicating them.
    """
    cold_start_root = Path(cold_start_root)
    if not cold_start_root.exists():
        return {}

    task_to_meta: Dict[str, Dict[str, Any]] = {}
    # Layout: <root>/<run_id>/<domain>/<task_uuid>/episode_*.json
    for run_dir in sorted(cold_start_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for domain_dir in sorted(run_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            if domain_filter and domain != domain_filter:
                continue
            for task_dir in sorted(domain_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                episodes = sorted(task_dir.glob("episode_*.json"))
                if not episodes:
                    continue
                # Use the first episode (deterministic re-runs).
                episode_path = episodes[0]
                try:
                    payload = json.loads(episode_path.read_text())
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "skip unreadable episode %s: %r", episode_path, exc,
                    )
                    continue
                metadata = payload.get("metadata") or {}
                task_id = metadata.get("task_id") or task_dir.name
                # Compose a canonical task_id that matches what the
                # transfer driver sees when it inspects ``ctx.state.task``.
                # The cold-start labeller writes domain.task_uuid as the
                # episode's `game_name`; the harness sees it as
                # state.task. Use both forms as keys so lookup works
                # regardless of which the dispatcher uses.
                full_id_a = f"{domain}.{task_id}"
                full_id_b = task_id
                first_action = ""
                experiences = payload.get("experiences") or []
                if experiences:
                    first_action = (experiences[0].get("action") or "").strip()
                frames_dir = task_dir / "frames" / "ep_000"
                meta = {
                    "domain": domain,
                    "task_id": task_id,
                    "episode_path": str(episode_path),
                    "frames_dir": (
                        str(frames_dir) if frames_dir.is_dir() else None
                    ),
                    "first_action": first_action,
                    "goal": payload.get("query") or payload.get("task") or "",
                }
                task_to_meta.setdefault(full_id_a, meta)
                task_to_meta.setdefault(full_id_b, meta)
    logger.info(
        "discovered %d task->osworld_meta mapping(s) under %s",
        len(task_to_meta), cold_start_root,
    )
    return task_to_meta


# ---------------------------------------------------------------------------
# Verb -> pyautogui code translation
# ---------------------------------------------------------------------------


def _verb_to_pyautogui_code(action_type: str, payload: Dict[str, Any]) -> Optional[str]:
    """Translate one (verb, payload) pair into a pyautogui code string.

    Returns ``None`` when the verb has no pyautogui translation
    (caller should fall through to the per-task stub). Conservative:
    only emits code for verbs whose payloads carry the necessary
    coordinates / keys; a verb with missing args returns ``None``
    rather than guessing.
    """
    op = action_type.upper()
    p = payload or {}

    if op == "CLICK":
        x = p.get("x"); y = p.get("y")
        if x is None or y is None:
            return None
        button = p.get("button", "left")
        clicks = int(p.get("clicks", 1))
        return f"pyautogui.click(x={int(x)}, y={int(y)}, button={button!r}, clicks={clicks})"

    if op == "DOUBLE_CLICK":
        x = p.get("x"); y = p.get("y")
        if x is None or y is None:
            return None
        return f"pyautogui.doubleClick(x={int(x)}, y={int(y)})"

    if op == "RIGHT_CLICK":
        x = p.get("x"); y = p.get("y")
        if x is None or y is None:
            return None
        return f"pyautogui.rightClick(x={int(x)}, y={int(y)})"

    if op == "TYPE":
        text = p.get("text") or p.get("value")
        if text is None:
            return None
        interval = float(p.get("interval", 0.0))
        return f"pyautogui.typewrite({str(text)!r}, interval={interval})"

    if op == "KEY_PRESS":
        key = p.get("key") or p.get("keys")
        if not key:
            return None
        return f"pyautogui.press({str(key)!r})"

    if op == "HOTKEY":
        keys = p.get("keys") or p.get("hotkey") or []
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split("+") if k.strip()]
        if not keys:
            return None
        joined = ", ".join(repr(str(k)) for k in keys)
        return f"pyautogui.hotkey({joined})"

    if op == "SCROLL":
        clicks = int(p.get("clicks", p.get("amount", -3)))
        x = p.get("x"); y = p.get("y")
        if x is not None and y is not None:
            return f"pyautogui.scroll({clicks}, x={int(x)}, y={int(y)})"
        return f"pyautogui.scroll({clicks})"

    if op == "MOVE_MOUSE":
        x = p.get("x"); y = p.get("y")
        if x is None or y is None:
            return None
        duration = float(p.get("duration", 0.0))
        return f"pyautogui.moveTo({int(x)}, {int(y)}, duration={duration})"

    if op == "WAIT":
        duration = float(p.get("duration", p.get("seconds", 0.5)))
        return f"time.sleep({duration})"

    if op in ("DONE", "FINISH"):
        # No-op on the desktop side; commit-only.
        return ""

    if op == "EXECUTE":
        # Authoring path: a hop with explicit code escapes our verb
        # taxonomy entirely.
        code = p.get("code") or p.get("action_code")
        if isinstance(code, str) and code.strip():
            return code

    return None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TaskAwareOsworldExecutor:
    """``HopExecutor`` wrapper with per-sample lazy executor binding for
    OSWorld.

    Suitable for the Stage 3 (osworld) dispatcher. Each transferred
    skill is evaluated against a different cold-start sample, and
    each sample pins to one container in the
    :class:`~harness._executor_helpers.osworld_client.OsworldContainerPool`
    (hash bucket per ``task_id``).
    """

    def __init__(
        self,
        task_to_meta: Dict[str, Dict[str, Any]],
        pool: Any,  # OsworldContainerPool, but lazy-typed to avoid hard import
        *,
        prefer_gdino: bool = False,
        confidence: float = 0.8,
    ) -> None:
        self._task_to_meta = dict(task_to_meta)
        self._pool = pool
        self._prefer_gdino = bool(prefer_gdino)
        self._confidence = float(confidence)
        # Real executor cache keyed by (container_port, screenshot_hash):
        # consecutive visual hops on the same desktop state reuse one
        # VisualReasoningExecutor build.
        self._real_cache: Dict[tuple, Any] = {}
        self._stub_cache: Dict[str, Any] = {}
        self._bare_stub: Optional[Any] = None

    def task_count(self) -> int:
        return len({m.get("task_id") for m in self._task_to_meta.values()})

    # ------------------------------------------------------------------
    # Stubs
    # ------------------------------------------------------------------

    def _bare_stub_executor(self) -> Any:
        if self._bare_stub is None:
            from harness.osworld_executor import make_osworld_executor
            self._bare_stub, _ = make_osworld_executor(domain="osworld")
        return self._bare_stub

    def _stub_for(self, meta: Dict[str, Any]) -> Any:
        """Per-task stub that carries ``meta`` in its evidence trail."""
        key = str(meta.get("task_id") or "")
        if key in self._stub_cache:
            return self._stub_cache[key]
        from harness.osworld_executor import make_osworld_executor
        executor, _ = make_osworld_executor(
            domain="osworld", task=str(meta.get("task_id") or ""),
        )
        self._stub_cache[key] = executor
        return executor

    # ------------------------------------------------------------------
    # Real-VLM executor for current screenshot
    # ------------------------------------------------------------------

    def _real_executor_for(self, client: Any) -> Optional[Any]:
        """Fetch the current screenshot from ``client`` and return a
        ``VisualReasoningExecutor`` cached on (port, screenshot_hash)."""
        try:
            png = client.screenshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "screenshot(:%d) failed (%r); falling back to stub",
                getattr(client, "port", 0), exc,
            )
            return None
        h = hashlib.md5(png).hexdigest()
        key = (getattr(client, "port", 0), h)
        if key in self._real_cache:
            return self._real_cache[key]
        try:
            from PIL import Image
            import io as _io
            from visual_reasoning_wrapper.skill_executor import (
                make_visual_reasoning_executor,
            )
            img = Image.open(_io.BytesIO(png))
            img.load()
            ex = make_visual_reasoning_executor(
                img,
                prefer_gdino=self._prefer_gdino,
                confidence=self._confidence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VisualReasoningExecutor build for osworld port %s failed (%r)",
                getattr(client, "port", "?"), exc,
            )
            self._real_cache[key] = None
            return None
        self._real_cache[key] = ex
        return ex

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
        meta = self._task_to_meta.get(task)
        if not meta or self._pool is None:
            return self._bare_stub_executor()(action_type, payload, ctx)

        op = (action_type or "").upper()
        client = self._pool.pin_for(task)

        # ── 1. Primitive desktop verb -- run pyautogui inside the container.
        if op in PRIMITIVE_DESKTOP_OPS or op == "EXECUTE":
            code = _verb_to_pyautogui_code(op, payload or {})
            if code is None:
                # Verb has no translation -- per-task stub keeps the
                # chain firing.
                return self._stub_for(meta)(action_type, payload, ctx)
            if not code.strip():
                # No-op (e.g. DONE / FINISH).
                return self._stub_for(meta)(action_type, payload, ctx)
            try:
                result = client.run_pyautogui(code, timeout_s=20.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "run_pyautogui on %s failed (%r); per-task stub",
                    client.name, exc,
                )
                return self._stub_for(meta)(action_type, payload, ctx)
            # Invalidate any cached real-executors for this container
            # since the screen state likely changed.
            self._invalidate_real_cache_for_port(getattr(client, "port", 0))
            from common.state_schema import EvidenceRef
            ev = EvidenceRef(
                source=f"osworld:{op.lower()}",
                locator=f"step={getattr(getattr(ctx, 'state', None), 'inner_step', 0)},"
                        f"container={client.name}",
                role="COMMIT" if op in ("DONE", "FINISH") else "GATHER",
                confidence=self._confidence,
                payload={
                    "container": client.name,
                    "port": client.port,
                    "code": code,
                    "status": result.get("status"),
                    "task_id": meta.get("task_id"),
                },
            )
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload or {}),
                    "code": code,
                    "container": client.name,
                    "result_status": result.get("status"),
                },
                "evidence": [ev],
            }

        # ── 2. InnerAction verb -- use the real visual executor.
        if op in INNER_ACTION_VERBS:
            real = self._real_executor_for(client)
            if real is None:
                return self._stub_for(meta)(action_type, payload, ctx)
            return real(action_type, payload, ctx)

        # ── 3. Anything else -- per-task stub.
        return self._stub_for(meta)(action_type, payload, ctx)

    def _invalidate_real_cache_for_port(self, port: int) -> None:
        keys = [k for k in self._real_cache if k[0] == port]
        for k in keys:
            self._real_cache.pop(k, None)
