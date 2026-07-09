"""BrowserGym-specific visual grounding, heuristics, VLM adapter, OmniParser
glue, and tools.

Code that only concerns BrowserGym (AXTree, ``set_of_marks``, browser
tabs, ``browsergym_id`` element ids) lives here so the cross-domain
schema and prompts in :mod:`vlm_wrapper.schema` can stay
environment-agnostic.

**Three heads in one place**

- :func:`browsergym_wrapper.heuristic.obs_to_schema` — AXTree → schema
  (fast, free, deterministic).
- :func:`browsergym_wrapper.adapter.generate_label` /
  :func:`browsergym_wrapper.adapter.browser_obs_to_schema` — screenshot
  → vision LLM → schema (GPT-4o by default; the canonical Head 2 path
  used by ``cascaded_ground``).
- :func:`browsergym_wrapper.grounding.grounding_obs_to_schema` /
  :func:`browsergym_wrapper.grounding.grounding_image_to_schema` —
  screenshot → OmniParser-v2 → schema (local Head 3).

**Tools** — :func:`browsergym_wrapper.tools.build_browser_registry`
exposes a :class:`vlm_wrapper.tools.ToolRegistry` of AXTree-backed
helpers (``query_element_bbox``, ``search_elements``,
``get_som_elements``, …) used by ``vlm_wrapper.tool_loop`` for
multi-turn grounding.

**Demos and fixtures** also live here so the BrowserGym package is
self-contained:

- :mod:`browsergym_wrapper.example` — synthetic shopping page (no
  BrowserGym install required).
- :mod:`browsergym_wrapper.example_grounding` — Head 3 demo (OmniParser-v2
  on a screenshot).
- :mod:`browsergym_wrapper.demo_visual_grounding` — full Mode A / B / C
  comparison driver.
- ``browsergym_wrapper/example_screenshot.png`` — synthetic shopping page
  reused by both demos and by ``vlm_wrapper.tests``.
- ``browsergym_wrapper/real_browser_*.png`` — captured live BrowserGym
  pages used by the schema-generation regression tests.

The legacy modules ``vlm_wrapper.browser_adapter``,
``vlm_wrapper.browser_heuristic``, ``vlm_wrapper.tools_browser``,
``vlm_wrapper.grounding_browsergym``, ``vlm_wrapper.example_browsergym``,
``vlm_wrapper.example_grounding`` and
``vlm_wrapper.demo_visual_grounding`` remain as thin re-export shims, so
existing imports keep working.
"""

from __future__ import annotations

from browsergym_wrapper.adapter import (
    browser_obs_to_schema,
    generate_label,
)
from browsergym_wrapper.heuristic import obs_to_schema as text_to_schema
from browsergym_wrapper.heuristic import obs_to_schema
from browsergym_wrapper.tools import build_browser_registry

__all__ = [
    "generate_label",
    "browser_obs_to_schema",
    "obs_to_schema",
    "text_to_schema",
    "build_browser_registry",
]

# OmniParser-backed grounding pulls in the heavyweight ``vlm_wrapper.grounding``
# (torch / transformers / OmniParser-v2). Make it optional so a thin install
# without the vision extras can still use the AXTree heuristic + the API-only
# vision head.
try:
    from browsergym_wrapper.grounding import (
        BBox,
        ScreenElement,
        grounding_image_to_schema,
        grounding_obs_to_schema,
    )

    __all__ += [
        "BBox",
        "ScreenElement",
        "grounding_image_to_schema",
        "grounding_obs_to_schema",
    ]
except ImportError:
    pass
