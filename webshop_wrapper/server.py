"""Boot/teardown helpers for the WebShop Flask server.

The agent-side env (typically ``browsergym``) and the WebShop env
(``webshop`` conda env, see ``install/install_webshop.sh``) cannot
share a single Python process — their dependency pins clash (Flask 2 +
gym 0.24 + NumPy 1.x in WebShop vs. modern gymnasium + NumPy 2.x in
browsergym).  We solve this the same way ``env_wrappers/subprocess_env.py``
solves it for Orak/gymv: spawn the server in a subprocess running its
own Python interpreter and communicate over the network (HTTP, port
3000 by default).

Three boot modes
----------------

1. **stub** — ``python -m webshop_wrapper.stub_app``.  No install
   required, runs in any env with Flask.  Good for unit tests + the
   AXTree smoke.

2. **full-local** — ``cd $WEBSHOP_DIR && python -m web_agent_site.app
   --port <port>`` using the ``webshop`` conda env's interpreter.
   Spawned by ``start_webshop_server(mode="full")``.

3. **external** — Server already running (e.g. systemd, docker).
   ``WebShopTask`` just connects to ``WEBSHOP_BASE_URL`` over HTTP.
   No subprocess managed by this module.

Usage::

    from webshop_wrapper.server import start_webshop_server

    handle = start_webshop_server(mode="stub", port=3000)
    try:
        # ... run agent ...
        pass
    finally:
        handle.stop()
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


# Default conda-env Python for the full-mode WebShop interpreter.
_DEFAULT_FULL_PYTHON = "/workspace/miniconda3/envs/webshop/bin/python"
_DEFAULT_WEBSHOP_DIR = "/workspace/WebShop"


@dataclass
class ServerHandle:
    """Process + connection info for a managed WebShop server."""

    proc: subprocess.Popen | None
    base_url: str
    mode: str
    pid: int | None = None

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        finally:
            self.proc = None

    def __del__(self) -> None:
        self.stop()


def _port_in_use(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _wait_for_health(base_url: str, timeout: float) -> bool:
    """Block until the server's ``/__bridge/session/fixed_0`` returns 200."""
    probe = base_url.rstrip("/") + "/__bridge/session/fixed_0"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(probe, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def start_webshop_server(
    mode: Literal["stub", "full", "external"] = "stub",
    *,
    host: str = "127.0.0.1",
    port: int = 3000,
    webshop_dir: str | None = None,
    python: str | None = None,
    startup_timeout: float = 60.0,
) -> ServerHandle:
    """Boot a WebShop server and wait for it to be reachable.

    Parameters
    ----------
    mode
        ``"stub"`` -> ``webshop_wrapper.stub_app`` (no install).
        ``"full"`` -> the real WebShop Flask app, spawned in the
        ``webshop`` conda env.
        ``"external"`` -> assume already running, just probe and return
        a handle with ``proc=None``.
    host, port
        Where the server binds (or where to reach an external server).
    webshop_dir
        Path to the cloned ``princeton-nlp/WebShop`` repo.  Defaults to
        ``$WEBSHOP_DIR`` env var or ``/workspace/WebShop``.
    python
        Interpreter to use for ``mode="full"``.  Defaults to
        ``/workspace/miniconda3/envs/webshop/bin/python``.
    startup_timeout
        How long to wait for the bridge endpoint to become reachable.
        Stub starts in <1 s; full mode can take 10-30 s on first boot
        (search-engine warm-up + spaCy import).

    Returns
    -------
    ServerHandle
        Call ``.stop()`` to terminate, or use as a context-managed
        resource (``__del__`` cleans up on GC).
    """
    base_url = f"http://{host}:{port}"

    if mode == "external":
        if not _wait_for_health(base_url, timeout=5.0):
            raise RuntimeError(
                f"mode=external but no healthy WebShop bridge at {base_url}"
            )
        logger.info("WebShop server: reusing external instance at %s", base_url)
        return ServerHandle(proc=None, base_url=base_url, mode=mode)

    if _port_in_use(host, port):
        # Probe the bridge endpoint; if it responds we can reuse.
        if _wait_for_health(base_url, timeout=2.0):
            logger.info(
                "WebShop server: port %d already in use and bridge is "
                "healthy — reusing", port,
            )
            return ServerHandle(proc=None, base_url=base_url, mode="external")
        raise RuntimeError(
            f"port {port} is in use but bridge at {base_url}/__bridge/... "
            f"is not responding — kill the offender or pick a different port"
        )

    if mode == "stub":
        cmd = [sys.executable, "-m", "webshop_wrapper.stub_app",
               "--host", host, "--port", str(port)]
        cwd = str(Path(__file__).resolve().parent.parent)
        env = os.environ.copy()
    elif mode == "full":
        wsdir = webshop_dir or os.environ.get("WEBSHOP_DIR", _DEFAULT_WEBSHOP_DIR)
        py = python or os.environ.get("WEBSHOP_PYTHON", _DEFAULT_FULL_PYTHON)
        if not Path(wsdir).is_dir():
            raise FileNotFoundError(
                f"WEBSHOP_DIR={wsdir} does not exist — run "
                f"`bash install/install_webshop.sh` first"
            )
        if not Path(py).is_file():
            raise FileNotFoundError(
                f"webshop interpreter {py} does not exist — run "
                f"`bash install/install_webshop.sh` first"
            )
        cmd = [py, "-m", "web_agent_site.app"]
        # web_agent_site/app.py hardcodes port=3000 + host=0.0.0.0; if
        # the caller asked for something different we set env vars and
        # rely on the ``--port`` patch from install_webshop.sh §5 which
        # also accepts ``WEBSHOP_PORT`` / ``WEBSHOP_HOST`` env overrides.
        env = os.environ.copy()
        env["WEBSHOP_HOST"] = host
        env["WEBSHOP_PORT"] = str(port)
        env["FLASK_RUN_HOST"] = host
        env["FLASK_RUN_PORT"] = str(port)
        cwd = wsdir
    else:
        raise ValueError(f"unknown mode {mode!r}")

    logger.info("Starting WebShop server (mode=%s) on %s ...", mode, base_url)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    if not _wait_for_health(base_url, timeout=startup_timeout):
        proc.terminate()
        try:
            stderr_tail = (proc.stderr.read() if proc.stderr else b"").decode(
                "utf-8", errors="replace",
            )[-2000:]
        except Exception:
            stderr_tail = ""
        raise RuntimeError(
            f"WebShop server did not become healthy at {base_url} "
            f"within {startup_timeout}s.\n--- stderr tail ---\n{stderr_tail}"
        )
    logger.info("WebShop server: ready at %s (pid=%d)", base_url, proc.pid)
    return ServerHandle(proc=proc, base_url=base_url, mode=mode, pid=proc.pid)


__all__ = ["ServerHandle", "start_webshop_server"]
