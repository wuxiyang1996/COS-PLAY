"""Per-sample lazy-binding executor wrapper for ``BrowserAdapter``.

Phase-5 §12.1 item 4 (Tier 1 BrowserGym real-env binding). Mirrors
the OSWorld pattern in :mod:`harness._osworld_per_sample_executor` but
uses a subprocess helper (``harness/_executor_helpers/browser_helper.py``)
running in the ``browsergym`` conda env, since BrowserGym + Playwright
are not in the harness's main env. The subprocess hosts one
``gym.Env`` instance per task and talks newline-delimited JSON-RPC
over stdin/stdout.

Routing rules per hop:

1. ``ctx.state.task`` not in map  ->  bare deterministic stub
   (``make_browsergym_executor()``).
2. Helper subprocess unreachable / failed to spawn  ->  bare
   deterministic stub (sticky for this executor's lifetime).
3. ``action_type`` outside :data:`INNER_ACTION_VERBS` and outside
   :data:`HIGH_LEVEL_BROWSER_OPS`  ->  per-task deterministic stub.
4. ``action_type`` in :data:`HIGH_LEVEL_BROWSER_OPS`  ->  translated
   to a BrowserGym high-level action string (e.g.
   ``'click("47")'``, ``'fill("17", "hello")'``, ``'noop()'``) and
   sent to the helper via ``op=step``.
5. ``action_type`` in :data:`INNER_ACTION_VERBS`  ->  current
   screenshot is fetched (cached) from the helper, a
   :class:`VisualReasoningExecutor` is built / cached against that
   screenshot, and the hop is dispatched to it.

Helper lifecycle: the helper is spawned lazily on the first hop that
actually needs it. Each new ``task_id`` triggers an ``op=start``
request to the helper which calls ``gym.make("browsergym/<task_id>")``
+ ``env.reset()`` and returns the initial obs. Subsequent hops on
that task reuse the same helper instance. When the executor goes
out of scope, the helper is told to ``close``.

The screenshot cache is keyed by ``(task_id, step_index,
screenshot_path)`` so consecutive visual hops on the same browser
state reuse one ``VisualReasoningExecutor`` build.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("harness.browser_per_sample_executor")

__all__ = [
    "HIGH_LEVEL_BROWSER_OPS",
    "INNER_ACTION_VERBS",
    "TaskAwareBrowserExecutor",
    "discover_task_to_browser_meta",
]


#: InnerAction verbs that the real :class:`VisualReasoningExecutor`
#: dispatches against the current screenshot.
INNER_ACTION_VERBS: frozenset[str] = frozenset({
    "GROUND", "RETRIEVE", "CHECK", "VERIFY", "COMMIT", "EXECUTE",
})

#: Verbs that translate to a BrowserGym high-level action string. The
#: high-level action set is documented at
#: :mod:`browsergym.core.action.highlevel`. ``EXECUTE`` is the
#: canonical InnerAction verb for "run the underlying primitive";
#: when a hop's payload carries an explicit ``action`` field we
#: forward it verbatim.
HIGH_LEVEL_BROWSER_OPS: frozenset[str] = frozenset({
    "CLICK", "FILL", "PRESS", "KEY_PRESS", "HOVER", "SCROLL",
    "SELECT_OPTION", "CHECK_BOX", "UNCHECK_BOX",
    "GOTO", "GO_BACK", "GO_FORWARD", "NEW_TAB", "TAB_FOCUS", "TAB_CLOSE",
    "NOOP", "DONE",
})


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_task_to_browser_meta(
    cold_start_root: Path,
    *,
    task_prefix: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Walk ``Cold-start-out-browsergym/<task_id>/episode_*.json`` and
    return ``{task_id: meta_dict}``.

    Each meta carries::

        {
            "task_id": "miniwob.email-inbox-star-reply",
            "browsergym_id": "browsergym/miniwob.email-inbox-star-reply",
            "episode_path": "Cold-start-out-browsergym/.../episode_000.json",
            "frames_dir": "Cold-start-out-browsergym/.../frames/ep_000",
            "first_action": 'click("47")',
            "goal": "Find the email by Cecile and click the star icon...",
        }

    The cold-start tree is FLAT under ``cold_start_root``: one dir
    per task_id, sample files directly inside. ``task_prefix`` (e.g.
    ``"miniwob"`` or ``"webarena"``) restricts the scan when the
    dispatcher only wants one family.
    """
    cold_start_root = Path(cold_start_root)
    if not cold_start_root.exists():
        return {}

    task_to_meta: Dict[str, Dict[str, Any]] = {}
    for task_dir in sorted(cold_start_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name
        if task_prefix and not task_id.startswith(task_prefix):
            continue
        episodes = sorted(task_dir.glob("episode_*.json"))
        if not episodes:
            continue
        episode_path = episodes[0]
        try:
            import json as _json
            payload = _json.loads(episode_path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "skip unreadable browser episode %s: %r", episode_path, exc,
            )
            continue
        first_action = ""
        experiences = payload.get("experiences") or []
        if experiences:
            first_action = (experiences[0].get("action") or "").strip()
        frames_dir = task_dir / "frames" / "ep_000"
        meta = {
            "task_id": task_id,
            "browsergym_id": (
                task_id if task_id.startswith("browsergym/")
                else f"browsergym/{task_id}"
            ),
            "episode_path": str(episode_path),
            "frames_dir": str(frames_dir) if frames_dir.is_dir() else None,
            "first_action": first_action,
            "goal": payload.get("query") or payload.get("task") or "",
        }
        task_to_meta[task_id] = meta
        # Also accept the "browsergym/<task_id>" form so dispatchers
        # that prefix the registry namespace match.
        task_to_meta[meta["browsergym_id"]] = meta
    logger.info(
        "discovered %d task->browser_meta mapping(s) under %s",
        len({m["task_id"] for m in task_to_meta.values()}, ),
        cold_start_root,
    )
    return task_to_meta


# ---------------------------------------------------------------------------
# Verb -> BrowserGym high-level action translation
# ---------------------------------------------------------------------------


def _verb_to_highlevel_action(action_type: str, payload: Dict[str, Any]) -> Optional[str]:
    """Translate (verb, payload) to a BrowserGym high-level action string.

    Returns ``None`` when the verb has no translation or required
    payload fields are missing -- caller should fall through to the
    per-task stub.
    """
    op = action_type.upper()
    p = payload or {}

    # Authoring escape hatch: a hop with explicit action= short-circuits.
    if op == "EXECUTE":
        action = p.get("action") or p.get("highlevel_action")
        if isinstance(action, str) and action.strip():
            return action

    bid = p.get("bid") or p.get("element_bid") or p.get("element_id")

    if op == "CLICK":
        if bid is None:
            return None
        button = p.get("button", "left")
        if button == "left":
            return f'click({str(bid)!r})'
        return f'click({str(bid)!r}, button={button!r})'

    if op == "FILL":
        if bid is None:
            return None
        text = p.get("text") or p.get("value") or ""
        return f'fill({str(bid)!r}, {str(text)!r})'

    if op in ("PRESS", "KEY_PRESS"):
        if bid is None:
            return None
        key = p.get("key") or ""
        if not key:
            return None
        return f'press({str(bid)!r}, {str(key)!r})'

    if op == "HOVER":
        if bid is None:
            return None
        return f'hover({str(bid)!r})'

    if op == "SCROLL":
        dx = int(p.get("dx", 0))
        dy = int(p.get("dy", p.get("amount", 300)))
        return f'scroll({dx}, {dy})'

    if op == "SELECT_OPTION":
        if bid is None:
            return None
        option = p.get("option") or p.get("value") or ""
        return f'select_option({str(bid)!r}, {str(option)!r})'

    if op == "CHECK_BOX":
        if bid is None:
            return None
        return f'check({str(bid)!r})'

    if op == "UNCHECK_BOX":
        if bid is None:
            return None
        return f'uncheck({str(bid)!r})'

    if op == "GOTO":
        url = p.get("url") or p.get("href") or ""
        if not url:
            return None
        return f'goto({str(url)!r})'

    if op == "GO_BACK":
        return "go_back()"
    if op == "GO_FORWARD":
        return "go_forward()"
    if op == "NEW_TAB":
        return "new_tab()"
    if op == "TAB_FOCUS":
        idx = int(p.get("index", 0))
        return f'tab_focus({idx})'
    if op == "TAB_CLOSE":
        return "tab_close()"
    if op == "NOOP":
        return "noop()"
    if op == "DONE":
        # Browser tasks don't have an explicit DONE action; map to
        # noop so the chain emits evidence and exits cleanly.
        return "noop()"

    return None


# ---------------------------------------------------------------------------
# Helper subprocess management
# ---------------------------------------------------------------------------


class _HelperSession:
    """One helper subprocess + the task it currently has loaded.

    The subprocess wraps a single ``gym.Env`` instance. To switch
    tasks we send ``op=start`` with the new task_id, which calls
    ``gym.make`` + ``env.reset()`` again (the helper's previous env
    is closed before).
    """

    def __init__(
        self,
        proc: subprocess.Popen,
        log_path: Path,
        *,
        spawn_lock: threading.Lock,
    ) -> None:
        self.proc = proc
        self.log_path = log_path
        self.current_task_id: str = ""
        self.last_obs: Dict[str, Any] = {}
        self.step_index: int = 0
        self._spawn_lock = spawn_lock

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class TaskAwareBrowserExecutor:
    """``HopExecutor`` with per-sample lazy executor binding for browser.

    Spawns one ``browser_helper`` subprocess in the ``browsergym``
    conda env on demand and reuses it across hops. Tasks are
    switched by sending ``op=start`` with a new task_id -- the
    helper closes its previous env and creates a new one.
    """

    def __init__(
        self,
        task_to_meta: Dict[str, Dict[str, Any]],
        *,
        conda_env: str = "browsergym",
        prefer_gdino: bool = False,
        confidence: float = 0.8,
        miniwob_url: Optional[str] = None,
        helper_log_dir: Optional[Path] = None,
        display: str = ":99",
    ) -> None:
        self._task_to_meta = dict(task_to_meta)
        self._conda_env = conda_env
        self._prefer_gdino = bool(prefer_gdino)
        self._confidence = float(confidence)
        self._miniwob_url = miniwob_url or (
            "file:///workspace/BrowserGym/miniwob-plusplus/miniwob/html/miniwob/"
        )
        self._display = display
        self._helper_log_dir = (
            Path(helper_log_dir) if helper_log_dir is not None
            else Path("/tmp")
        )
        self._spawn_lock = threading.Lock()
        self._session: Optional[_HelperSession] = None
        self._spawn_failed = False
        self._real_cache: Dict[Tuple[str, int, str], Any] = {}
        self._stub_cache: Dict[str, Any] = {}
        self._bare_stub: Optional[Any] = None

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def task_count(self) -> int:
        return len({m.get("task_id") for m in self._task_to_meta.values()})

    # ------------------------------------------------------------------
    # Helper management
    # ------------------------------------------------------------------

    def _ensure_helper(self) -> Optional[_HelperSession]:
        """Lazily spawn the helper. Sticky failure: once spawn fails,
        every subsequent call returns ``None`` (the per-task stub
        fallback path)."""
        if self._spawn_failed:
            return None
        if self._session is not None and self._session.is_alive():
            return self._session
        with self._spawn_lock:
            if self._session is not None and self._session.is_alive():
                return self._session
            try:
                from harness._executor_helpers._proto import spawn_helper
                helper_path = (
                    Path(__file__).resolve().parent
                    / "_executor_helpers" / "browser_helper.py"
                )
                log = self._helper_log_dir / f"_browser_helper_{os.getpid()}.stderr.log"
                proc = spawn_helper(
                    helper_path,
                    conda_env=self._conda_env,
                    extra_env={
                        "DISPLAY": self._display,
                        "MINIWOB_URL": self._miniwob_url,
                    },
                    log_stderr_to=log,
                )
                self._session = _HelperSession(
                    proc, log, spawn_lock=self._spawn_lock,
                )
                logger.info(
                    "spawned browser helper in conda env %s (log=%s)",
                    self._conda_env, log,
                )
                return self._session
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "failed to spawn browser helper in env %r: %r; "
                    "falling back to per-task stub for the rest of "
                    "this dispatcher's lifetime",
                    self._conda_env, exc,
                )
                self._spawn_failed = True
                self._session = None
                return None

    def _ensure_task_loaded(
        self, session: _HelperSession, browsergym_id: str,
    ) -> bool:
        """Send ``op=start`` if the helper isn't already on this task."""
        if session.current_task_id == browsergym_id:
            return True
        from harness._executor_helpers._proto import rpc_call, RPCError
        try:
            resp = rpc_call(
                session.proc, "start",
                {"task_id": browsergym_id, "headless": True, "axtree_chars": 0},
                timeout_s=120.0,
            )
        except RPCError as exc:
            logger.warning(
                "browser helper start(%s) failed: %r", browsergym_id, exc,
            )
            return False
        session.current_task_id = browsergym_id
        session.last_obs = resp
        session.step_index = int(resp.get("step_index", 0))
        return True

    def shutdown(self) -> None:
        if self._session is None:
            return
        from harness._executor_helpers._proto import rpc_close
        try:
            rpc_close(self._session.proc, timeout_s=5.0)
        except Exception:  # noqa: BLE001
            pass
        self._session = None

    # ------------------------------------------------------------------
    # Stubs
    # ------------------------------------------------------------------

    def _bare_stub_executor(self) -> Any:
        if self._bare_stub is None:
            from harness.browsergym_executor import make_browsergym_executor
            self._bare_stub, _ = make_browsergym_executor(
                domain="browser", task="",
            )
        return self._bare_stub

    def _stub_for(self, meta: Dict[str, Any]) -> Any:
        key = str(meta.get("task_id") or "")
        if key in self._stub_cache:
            return self._stub_cache[key]
        from harness.browsergym_executor import make_browsergym_executor
        executor, _ = make_browsergym_executor(
            domain="browser", task=str(meta.get("task_id") or ""),
        )
        self._stub_cache[key] = executor
        return executor

    # ------------------------------------------------------------------
    # Real visual executor (cached on screenshot)
    # ------------------------------------------------------------------

    def _real_executor_for(
        self, session: _HelperSession, browsergym_id: str,
    ) -> Optional[Any]:
        sshot_path = session.last_obs.get("screenshot_path")
        if not sshot_path or not Path(sshot_path).is_file():
            return None
        key = (browsergym_id, session.step_index, sshot_path)
        if key in self._real_cache:
            return self._real_cache[key]
        try:
            from PIL import Image
            from visual_reasoning_wrapper.skill_executor import (
                make_visual_reasoning_executor,
            )
            img = Image.open(sshot_path)
            img.load()
            ex = make_visual_reasoning_executor(
                img,
                prefer_gdino=self._prefer_gdino,
                confidence=self._confidence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "VisualReasoningExecutor build for browser %s failed (%r)",
                browsergym_id, exc,
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
        if not meta:
            return self._bare_stub_executor()(action_type, payload, ctx)

        op = (action_type or "").upper()
        is_action = op in HIGH_LEVEL_BROWSER_OPS or op == "EXECUTE"
        is_visual = op in INNER_ACTION_VERBS

        if not (is_action or is_visual):
            return self._stub_for(meta)(action_type, payload, ctx)

        session = self._ensure_helper()
        if session is None:
            return self._stub_for(meta)(action_type, payload, ctx)
        if not self._ensure_task_loaded(session, meta["browsergym_id"]):
            return self._stub_for(meta)(action_type, payload, ctx)

        # ── Action verb -- send through the helper.
        if is_action:
            action_str = _verb_to_highlevel_action(op, payload or {})
            if action_str is None or not action_str.strip():
                return self._stub_for(meta)(action_type, payload, ctx)
            from harness._executor_helpers._proto import rpc_call, RPCError
            try:
                resp = rpc_call(
                    session.proc, "step",
                    {"action": action_str, "axtree_chars": 0},
                    timeout_s=30.0,
                )
            except RPCError as exc:
                logger.warning(
                    "browser helper step(%s) failed: %r", action_str, exc,
                )
                return self._stub_for(meta)(action_type, payload, ctx)
            session.last_obs = resp
            session.step_index = int(resp.get("step_index", session.step_index + 1))
            from common.state_schema import EvidenceRef
            ev = EvidenceRef(
                source=f"browser:{op.lower()}",
                locator=(
                    f"step={getattr(getattr(ctx, 'state', None), 'inner_step', 0)},"
                    f"task={meta['task_id']}"
                ),
                role="COMMIT" if op in ("DONE",) else "GATHER",
                confidence=self._confidence,
                payload={
                    "task_id": meta["task_id"],
                    "highlevel_action": action_str,
                    "url": resp.get("url"),
                    "reward": resp.get("reward"),
                    "terminated": resp.get("terminated"),
                    "last_action_error": resp.get("last_action_error"),
                },
            )
            return {
                "ok": True,
                "observation": {
                    "echo_action": op,
                    "echo_payload": dict(payload or {}),
                    "highlevel_action": action_str,
                    "url": resp.get("url"),
                    "reward": resp.get("reward"),
                    "terminated": resp.get("terminated"),
                },
                "evidence": [ev],
            }

        # ── InnerAction visual verb -- dispatch to VisualReasoning.
        if is_visual:
            real = self._real_executor_for(session, meta["browsergym_id"])
            if real is None:
                return self._stub_for(meta)(action_type, payload, ctx)
            return real(action_type, payload, ctx)

        # Should be unreachable.
        return self._stub_for(meta)(action_type, payload, ctx)
