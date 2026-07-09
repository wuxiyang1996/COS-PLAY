"""HTTP client for the OSWorld desktop server running inside a
``happysixd/osworld-docker`` container.

Unlike :mod:`harness._executor_helpers.browser_helper` (which is a
subprocess living in the ``browsergym`` conda env), the OSWorld
container *itself* is the subprocess -- it runs a Flask HTTP server
on its internal port 5000 (Docker-mapped to a host port), exposing
``/screenshot`` / ``/execute`` / ``/run_python`` / ``/accessibility``
/ ``/cursor_position`` / ``/platform``. So the per-sample executor
just talks to it via HTTP from the main env, no extra translation
layer needed.

This module wraps that HTTP surface with a small typed client and
adds:

* :func:`discover_running_containers` -- enumerates docker
  containers built from ``happysixd/osworld-docker`` and returns
  their host-mapped port-5000 ports. The 13-container fleet that
  ships pre-warmed in this workspace is the canonical input.
* :class:`OsworldContainerPool` -- round-robin load balancer over a
  pool of host ports so concurrent skills get spread across the
  fleet rather than serialised on one container.
* :class:`OsworldClient` -- a single-port client. ``screenshot()``,
  ``run_pyautogui(code)``, ``a11y_tree()``, ``platform()``,
  ``cursor_position()``. Each call has a configurable timeout.

The client is deliberately stateless about which task is "loaded"
in the container -- the cross-domain transfer measurement does not
need a fresh task snapshot per episode (we are evaluating whether
a *transferred* skill grounds against a real desktop, not whether
it solves the canonical OSWorld task). If a future caller wants the
full ``DesktopEnv.reset(task_config=...)`` lifecycle, they can layer
that on top.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("harness.executor_helpers.osworld_client")

__all__ = [
    "OsworldClient",
    "OsworldContainerPool",
    "OsworldHTTPError",
    "discover_running_containers",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OsworldHTTPError(RuntimeError):
    """Raised on any non-2xx response from the OSWorld container."""

    def __init__(self, message: str, status_code: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


# ---------------------------------------------------------------------------
# Discovery -- enumerate live containers
# ---------------------------------------------------------------------------


def discover_running_containers(
    *,
    image: str = "happysixd/osworld-docker",
    timeout_s: float = 5.0,
) -> List[Tuple[str, int]]:
    """Return ``[(container_name, host_port_for_5000), ...]`` for every
    running container built from ``image``.

    Uses ``docker ps`` with a structured format string so we don't
    have to depend on the ``docker`` python SDK. Returns an empty
    list when the daemon is unreachable or no matching containers
    are running -- callers should treat that as "fall back to the
    deterministic stub" (see
    :class:`harness._osworld_per_sample_executor.TaskAwareOsworldExecutor`).

    Container ports are parsed from ``{{.Ports}}`` which looks like
    ``0.0.0.0:5039->5000/tcp, 0.0.0.0:8044->8006/tcp, ...``. We
    only care about the entry that maps host -> container 5000.
    """
    try:
        out = subprocess.check_output(
            ["docker", "ps",
             "--filter", f"ancestor={image}",
             "--format", "{{.Names}}|{{.Ports}}"],
            timeout=timeout_s, text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError) as exc:
        logger.warning(
            "discover_running_containers: docker ps failed (%r); "
            "returning empty pool", exc,
        )
        return []

    pool: List[Tuple[str, int]] = []
    for line in out.strip().splitlines():
        if "|" not in line:
            continue
        name, ports = line.split("|", 1)
        host_port = _parse_host_port_for(ports, container_port=5000)
        if host_port is None:
            continue
        pool.append((name.strip(), host_port))
    return pool


def _parse_host_port_for(ports_str: str, *, container_port: int) -> Optional[int]:
    """Pull host-side port number out of a docker-ps ports column.

    Format example::

        0.0.0.0:5039->5000/tcp, [::]:5039->5000/tcp, 22/tcp, 5900/tcp

    We want the host port that maps to ``container_port`` (5000 for
    the OSWorld API server).
    """
    needle = f"->{container_port}/"
    for chunk in ports_str.split(","):
        chunk = chunk.strip()
        if needle not in chunk:
            continue
        # Take the first ``host:port->container/proto`` form.
        try:
            host_part, _ = chunk.split("->", 1)
            # host_part is like "0.0.0.0:5039" or "[::]:5039"
            return int(host_part.rsplit(":", 1)[-1])
        except (ValueError, IndexError):
            continue
    return None


# ---------------------------------------------------------------------------
# Client -- one host port
# ---------------------------------------------------------------------------


@dataclass
class OsworldClient:
    """Typed HTTP client for one OSWorld container.

    ``host`` defaults to ``localhost`` since this workspace's
    containers expose their port-5000 on ``0.0.0.0`` of the host.
    Cross-machine setups should pass an explicit hostname.
    """

    port: int
    host: str = "localhost"
    name: str = ""
    timeout_s: float = 30.0
    pkgs_prefix: str = (
        "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"
    )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ------------------------------------------------------------------
    # Probes / observations
    # ------------------------------------------------------------------

    def screenshot(self, *, save_to: Optional[Path] = None) -> bytes:
        """Return the raw PNG bytes; optionally also write to disk."""
        body, _ = self._get("/screenshot", expect_binary=True)
        if save_to is not None:
            Path(save_to).write_bytes(body)
        return body

    def screenshot_pil(self) -> "Any":
        """Convenience: return a PIL.Image. Imports PIL lazily."""
        from PIL import Image
        return Image.open(io.BytesIO(self.screenshot()))

    def a11y_tree(self) -> str:
        """Return the AT-SPI accessibility-tree XML/text dump.

        The OSWorld server returns this as a UTF-8 text body. Empty
        string when AT-SPI is not initialised inside the container.
        """
        body, _ = self._get("/accessibility", expect_binary=False)
        if isinstance(body, bytes):
            return body.decode("utf-8", errors="replace")
        return str(body)

    def cursor_position(self) -> Tuple[int, int]:
        body, _ = self._get("/cursor_position", expect_binary=False)
        try:
            arr = json.loads(body) if isinstance(body, str) else body
            return int(arr[0]), int(arr[1])
        except Exception:  # noqa: BLE001
            return (0, 0)

    def platform(self) -> str:
        body, _ = self._get("/platform", expect_binary=False)
        return body.strip() if isinstance(body, str) else ""

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def run_pyautogui(
        self,
        code: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run a pyautogui snippet (or any python code) inside the container.

        Wraps ``code`` with ``import pyautogui; pyautogui.FAILSAFE = False; ...``
        and POSTs to ``/run_python`` (which returns structured JSON,
        unlike ``/execute`` which forks a python subprocess on the
        server side and returns stdout/stderr). On non-2xx returns
        :class:`OsworldHTTPError`; on 2xx returns the parsed JSON.
        """
        url = f"{self.base_url}/run_python"
        wrapped = self.pkgs_prefix.format(command=code)
        return self._post(
            url, payload={"code": wrapped},
            timeout_s=timeout_s if timeout_s is not None else self.timeout_s,
        )

    def run_python_script(
        self,
        script: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run an arbitrary python script (no pyautogui prefix wrap)."""
        url = f"{self.base_url}/run_python"
        return self._post(
            url, payload={"code": script},
            timeout_s=timeout_s if timeout_s is not None else self.timeout_s,
        )

    # ------------------------------------------------------------------
    # Internal HTTP plumbing
    # ------------------------------------------------------------------

    def _get(
        self, path: str, *, expect_binary: bool,
    ) -> Tuple[Any, Dict[str, str]]:
        import requests
        url = f"{self.base_url}{path}"
        try:
            resp = requests.get(url, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise OsworldHTTPError(
                f"GET {url} failed: {type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise OsworldHTTPError(
                f"GET {url} returned {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text[:300],
            )
        if expect_binary:
            return resp.content, dict(resp.headers)
        return resp.text, dict(resp.headers)

    def _post(
        self, url: str, *, payload: Dict[str, Any], timeout_s: float,
    ) -> Dict[str, Any]:
        import requests
        try:
            resp = requests.post(
                url, json=payload, timeout=timeout_s,
                headers={"Content-Type": "application/json"},
            )
        except requests.RequestException as exc:
            raise OsworldHTTPError(
                f"POST {url} failed: {type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code != 200:
            raise OsworldHTTPError(
                f"POST {url} returned {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text[:300],
            )
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise OsworldHTTPError(
                f"POST {url} returned non-JSON body: {resp.text[:200]!r}",
                status_code=resp.status_code,
                body=resp.text[:300],
            ) from exc


# ---------------------------------------------------------------------------
# Pool -- round-robin balancer
# ---------------------------------------------------------------------------


class OsworldContainerPool:
    """Round-robin load balancer over a fleet of OSWorld containers.

    Constructed with the output of :func:`discover_running_containers`
    (or an explicit list of ``(name, port)`` pairs for tests).

    ``pin_for(task_id)`` returns a stable client per task_id (hash
    bucket) so a hot loop of hops on the same task keeps hitting the
    same container -- useful when one container's state diverges
    from another's (different open windows, different cursor
    positions). ``next_round_robin()`` picks any container; useful
    for stateless ops like screenshot probes during smoke tests.
    """

    def __init__(
        self,
        members: List[Tuple[str, int]],
        *,
        host: str = "localhost",
        timeout_s: float = 30.0,
    ) -> None:
        if not members:
            raise ValueError("OsworldContainerPool: members list is empty")
        self._members = list(members)
        self._host = host
        self._timeout_s = timeout_s
        self._idx_lock = threading.Lock()
        self._next_idx = 0
        self._client_cache: Dict[int, OsworldClient] = {}

    @classmethod
    def from_discovery(
        cls,
        *,
        image: str = "happysixd/osworld-docker",
        host: str = "localhost",
        timeout_s: float = 30.0,
    ) -> Optional["OsworldContainerPool"]:
        """Discover and return a pool, or ``None`` when none are running."""
        members = discover_running_containers(image=image)
        if not members:
            return None
        return cls(members, host=host, timeout_s=timeout_s)

    @property
    def size(self) -> int:
        return len(self._members)

    def _client_for(self, name: str, port: int) -> OsworldClient:
        if port in self._client_cache:
            return self._client_cache[port]
        c = OsworldClient(
            port=port, host=self._host, name=name, timeout_s=self._timeout_s,
        )
        self._client_cache[port] = c
        return c

    def next_round_robin(self) -> OsworldClient:
        with self._idx_lock:
            i = self._next_idx % len(self._members)
            self._next_idx += 1
        name, port = self._members[i]
        return self._client_for(name, port)

    def pin_for(self, task_id: str) -> OsworldClient:
        """Stable hash-bucket assignment so each task_id pins to one
        container. Falls through to ``next_round_robin`` for empty
        task_id (caller should not rely on that path)."""
        if not task_id:
            return self.next_round_robin()
        bucket = hash(task_id) % len(self._members)
        name, port = self._members[bucket]
        return self._client_for(name, port)
