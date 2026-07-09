"""Subprocess helpers for cross-env hop execution.

The harness's main pipeline runs from the same Python interpreter that
imports ``cv2`` / ``decord`` / the visual_reasoning_wrapper stack
(``base`` conda env). Some target executors (BrowserGym + Playwright)
need a different package set that lives in a separate conda env. The
helpers in this package run inside the target env and talk
newline-delimited JSON-RPC over stdin/stdout to the per-sample
executor in the main env.

Files:

* :mod:`_proto` -- shared JSON-RPC framing + ``spawn_helper`` factory.
* :mod:`browser_helper` -- BrowserGym ``gym.Env`` wrapper, runs in the
  ``browsergym`` conda env. Spawned by
  :class:`harness._browser_per_sample_executor.TaskAwareBrowserExecutor`.

Note: the OSWorld real-env executor does NOT use a subprocess helper.
The OSWorld Docker containers (``happysixd/osworld-docker``) expose
their own Flask HTTP server (ports ``/execute`` / ``/screenshot`` /
``/accessibility``), so
:class:`harness._osworld_per_sample_executor.TaskAwareOsworldExecutor`
talks to them directly over HTTP rather than spawning yet another
translation layer.
"""

__all__: list[str] = []
