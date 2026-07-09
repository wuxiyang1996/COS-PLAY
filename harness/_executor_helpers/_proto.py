"""Shared JSON-RPC framing + ``spawn_helper`` factory for subprocess
helpers that run inside a different conda env than the harness's main
process.

Wire format
-----------
Newline-delimited JSON over stdin/stdout. One request, one response,
in order. No interleaving. No streaming. The request and response are
both JSON objects on a single line terminated with ``\\n``::

    >>> # client side
    >>> proc.stdin.write(json.dumps({"op": "step", ...}) + "\\n")
    >>> proc.stdin.flush()
    >>> resp = json.loads(proc.stdout.readline())

The helper process reads one line at a time, dispatches on
``request["op"]``, and writes one line back on stdout. Anything
written to stderr is logged by the parent but does not change the
protocol state, so debug prints from the helper are safe.

Response shape
--------------
Always a JSON object with at minimum ``{"ok": bool}``. On ``ok=True``
the response carries op-specific fields; on ``ok=False`` it carries
``{"error": str, "tb": Optional[str]}``. The helper never raises out
of the dispatch loop -- exceptions are caught and turned into
``ok=False`` responses so the parent's read never blocks on a missing
line.

Helpers handle these standard ops:

* ``ping``       -- liveness check, returns ``{"ok": true, "pid": int}``.
* ``close``      -- gracefully tear down the helper's owned state.

Plus per-helper ops (e.g. ``start`` / ``step`` / ``screenshot`` for
``browser_helper``).

Spawn convention
----------------
:func:`spawn_helper` shells out to ``conda run -n <env> python <script>``
which is the only conda invocation that reliably preserves the env's
``site-packages`` path inside subprocesses across all conda versions
without sourcing ``activate`` (which doesn't compose with non-bash
parent shells). The helper script must therefore be importable as a
standalone Python file -- no relative-import hacks, only absolute
imports of packages installed in the target env.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("harness.executor_helpers.proto")

__all__ = [
    "RPCError",
    "rpc_call",
    "rpc_close",
    "rpc_ping",
    "run_helper_loop",
    "spawn_helper",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RPCError(RuntimeError):
    """Raised when a helper subprocess returns ``{"ok": false}`` or
    when the wire protocol is violated (missing line, malformed JSON,
    helper died mid-call). The ``payload`` attribute carries the raw
    response dict when available so callers can introspect.
    """

    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


# ---------------------------------------------------------------------------
# Client-side helpers
# ---------------------------------------------------------------------------


def rpc_call(
    proc: subprocess.Popen,
    op: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    timeout_s: float = 60.0,
) -> Dict[str, Any]:
    """Send one ``{"op": op, ...payload}`` request and read one response.

    Raises :class:`RPCError` on any wire-protocol violation, helper
    exception, or response timeout. The helper subprocess must be
    spawned with ``stdin=PIPE, stdout=PIPE`` (see :func:`spawn_helper`).

    The ``timeout_s`` parameter only bounds the *response read*, not
    the helper's compute time -- if the helper is busy, this will
    block until it produces a line or the OS pipe is closed. Use a
    watchdog at the caller layer for hard time limits.
    """
    if proc.poll() is not None:
        raise RPCError(
            f"helper subprocess already exited (rc={proc.returncode}); "
            "cannot call op={op!r}"
        )
    msg: Dict[str, Any] = {"op": op}
    if payload:
        msg.update(payload)
    line = json.dumps(msg, default=str) + "\n"
    try:
        assert proc.stdin is not None
        proc.stdin.write(line)
        proc.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        raise RPCError(f"helper stdin closed: {exc!r}") from exc

    deadline = time.monotonic() + max(0.1, float(timeout_s))
    assert proc.stdout is not None
    while True:
        if proc.poll() is not None and time.monotonic() > deadline:
            raise RPCError(
                f"helper exited before responding (rc={proc.returncode})"
            )
        if time.monotonic() > deadline:
            raise RPCError(f"helper timed out after {timeout_s}s on op={op!r}")
        # Reading line-by-line; readline() blocks until \n or EOF.
        raw = proc.stdout.readline()
        if not raw:
            # EOF on stdout -- helper closed it (probably crashed).
            raise RPCError(
                f"helper closed stdout on op={op!r} "
                f"(rc={proc.returncode})"
            )
        raw = raw.strip()
        if not raw:
            continue
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RPCError(
                f"helper emitted non-JSON line on op={op!r}: {raw[:200]!r}"
            ) from exc
        if not isinstance(resp, dict):
            raise RPCError(
                f"helper emitted non-object response on op={op!r}: {raw[:200]!r}"
            )
        if not resp.get("ok"):
            err = resp.get("error") or "unknown_helper_error"
            raise RPCError(
                f"helper op={op!r} failed: {err}",
                payload=resp,
            )
        return resp


def rpc_ping(proc: subprocess.Popen, *, timeout_s: float = 10.0) -> int:
    """Return helper PID if reachable; raise :class:`RPCError` otherwise."""
    resp = rpc_call(proc, "ping", timeout_s=timeout_s)
    return int(resp.get("pid", -1))


def rpc_close(proc: subprocess.Popen, *, timeout_s: float = 15.0) -> None:
    """Send ``close`` then wait for graceful exit; SIGTERM/SIGKILL as fallback."""
    try:
        rpc_call(proc, "close", timeout_s=timeout_s)
    except RPCError as exc:
        logger.debug("rpc_close call failed (likely fine on shutdown): %r", exc)
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Spawn factory
# ---------------------------------------------------------------------------


def spawn_helper(
    helper_script: Path,
    *,
    conda_env: str,
    extra_env: Optional[Dict[str, str]] = None,
    extra_args: Optional[List[str]] = None,
    log_stderr_to: Optional[Path] = None,
) -> subprocess.Popen:
    """Spawn ``conda run -n <conda_env> python <helper_script>`` with
    pipes wired for newline-delimited JSON-RPC.

    The returned :class:`subprocess.Popen` is line-buffered (text-mode
    UTF-8). Stderr is either drained to ``log_stderr_to`` (file path)
    or inherited from the parent so debug prints surface in the
    harness logs.

    Pings the helper via :func:`rpc_ping` before returning to surface
    spawn failures (env-not-found, missing script, import error) early
    rather than at the first ``op=start`` call.

    Raises :class:`RPCError` if the helper does not respond to ``ping``
    within ``ping_timeout_s`` seconds (default 30).
    """
    helper_script = Path(helper_script)
    if not helper_script.is_file():
        raise FileNotFoundError(f"helper script not found: {helper_script}")

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "-u", str(helper_script),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # Force UTF-8 output regardless of locale so JSON containing
    # non-ASCII (BrowserGym goal text, OSWorld a11y labels) survives.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LANG", "C.UTF-8")

    stderr = (
        open(log_stderr_to, "ab") if log_stderr_to is not None
        else sys.stderr
    )
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr,
        env=env,
        text=True,
        bufsize=1,  # line-buffered
    )

    # Liveness probe: wait up to 30s for the helper to import its
    # heavy deps (browsergym + playwright is the slow case at ~2-3s
    # cold). Failure here usually means the env is missing or the
    # script raised at import time.
    try:
        pid = rpc_ping(proc, timeout_s=30.0)
        logger.info(
            "spawned helper %s in conda env %s (pid=%d)",
            helper_script.name, conda_env, pid,
        )
        return proc
    except RPCError:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


# ---------------------------------------------------------------------------
# Helper-side dispatch loop
# ---------------------------------------------------------------------------


def run_helper_loop(
    handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    *,
    name: str = "helper",
) -> None:
    """Run the standard read-line / dispatch / write-line loop.

    ``handlers`` is a ``{op_name: callable(request_dict) -> response_dict}``
    mapping. The callable should NOT raise on user-facing errors; it
    should return ``{"ok": False, "error": "..."}`` instead. The loop
    itself catches any exception that escapes (including from
    ``handlers["close"]``) and converts it to an ``{"ok": False,
    "error": ..., "tb": ...}`` response, then continues -- only EOF
    on stdin terminates the loop.

    Standard ops are auto-injected if not present in ``handlers``:

    * ``ping`` -> ``{"ok": True, "pid": os.getpid()}``
    * ``close`` -> ``{"ok": True}`` (caller is expected to register a
      real ``close`` handler that tears down state; this stub is only
      a fallback so a crash before registration doesn't deadlock).
    """
    if "ping" not in handlers:
        handlers = {
            **handlers,
            "ping": lambda _req: {"ok": True, "pid": os.getpid()},
        }
    if "close" not in handlers:
        handlers = {
            **handlers,
            "close": lambda _req: {"ok": True},
        }

    logger.info("[%s] entering RPC loop (pid=%d)", name, os.getpid())
    while True:
        line = sys.stdin.readline()
        if not line:
            logger.info("[%s] stdin EOF, exiting", name)
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({
                "ok": False,
                "error": f"invalid_json: {exc!r}",
                "raw_excerpt": line[:200],
            }) + "\n")
            sys.stdout.flush()
            continue
        op = str(req.get("op", "")).strip()
        handler = handlers.get(op)
        if handler is None:
            sys.stdout.write(json.dumps({
                "ok": False,
                "error": f"unknown_op: {op!r}",
                "known_ops": sorted(handlers),
            }) + "\n")
            sys.stdout.flush()
            continue
        try:
            resp = handler(req)
            if not isinstance(resp, dict):
                resp = {"ok": False, "error": f"handler returned non-dict: {type(resp).__name__}"}
            elif "ok" not in resp:
                resp = {"ok": True, **resp}
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            resp = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "tb": traceback.format_exc(),
            }
        sys.stdout.write(json.dumps(resp, default=str) + "\n")
        sys.stdout.flush()
        # If the handler signalled close, stop reading.
        if op == "close" and resp.get("ok"):
            logger.info("[%s] close received, exiting", name)
            break
