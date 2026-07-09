"""WebShop wrapper — bridges princeton-nlp/WebShop into the BrowserGym pipeline.

WebShop (Yao et al. 2022) is a single-Flask-server simulated shopping
benchmark.  Compared to WebArena / VisualWebArena, it has dramatically
simpler infra (one Flask process, rule-based reward, no LLM judge, no
Docker fleet — see `legacy/visualwebarena/README.md` for the contrast),
but it does **not** natively expose AXTree / `bid` / `extra_element_properties`.

**Strategy** — point ``browsergym/openended`` at the running WebShop Flask
server and let BrowserGym's native Chromium-based AXTree extraction do the
work.  The WebShop HTML templates use plain semantic markup
(``<button type="submit">``, ``<input type="radio">``, ``<a href>``, etc.)
which Chromium parses into clean accessibility roles, so all three
``browsergym_wrapper`` heads (heuristic / vision / OmniParser) reuse
out of the box.

**Two run modes**:

1. **Stub mode** — ``stub_app.py`` serves WebShop's *real* HTML templates
   with a tiny in-memory fake-product set.  Zero install (only Flask).
   Use to verify AXTree extraction quality before paying the full WebShop
   install cost.  See ``smoke_axtree.py``.

2. **Full mode** — ``install/install_webshop.sh`` clones princeton-nlp/WebShop
   into a dedicated conda env, downloads the small (1k-product) dataset,
   and patches the search engine to a BM25-only fallback so we don't need
   pyserini + Java.  ``server.py`` boots it as a subprocess.

The bridge module ``task.py`` registers a ``WebShopTask(AbstractBrowserTask)``
under ``browsergym/webshop.<goal_idx>`` so the existing
``cold_start/run_coldstart_actor_browsergym.sh`` driver and all 116
anti-thrash / anti-repeat / search-first regression tests continue to
work without modification.
"""

from __future__ import annotations

# Public API mirrors the browsergym_wrapper layout.  Lazy-import server.py
# so importing this package does not require Flask in the agent-side env.
from webshop_wrapper.task import WebShopTask, register_webshop_tasks

# Auto-register browsergym/webshop.<idx> on import — mirrors what
# ``browsergym/assistantbench/__init__.py`` does for AssistantBench
# tasks.  The driver in
# ``cold_start/generate_cold_start_actor_browsergym.py`` discovers the
# tasks by walking ``gymnasium.envs.registry`` after best-effort
# importing each entry in ``_OPTIONAL_TASK_SUITE_MODULES`` (which
# includes ``"webshop_wrapper"`` once this module is added there).
register_webshop_tasks()

__all__ = [
    "WebShopTask",
    "register_webshop_tasks",
]
