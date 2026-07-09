"""OSWorld-specific visual grounding, VLM adapter, OmniParser glue, and tools.

Code that only concerns [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld)
(``pyautogui``-style actions, OS-level accessibility trees, terminal
context) lives here so the cross-domain code in :mod:`vlm_wrapper` can
stay environment-agnostic.

**Three heads** — each independently produces a ``<state>...</state>``
schema for the ``desktop`` domain, so ``vlm_wrapper.cascaded_ground``
can cross-check them or escalate when one disagrees.

- :func:`osworld_wrapper.heuristic.obs_to_schema` /
  :func:`osworld_wrapper.heuristic.xml_to_schema` — Head 1: deterministic
  walker over the namespaced AT-SPI / UI-Automation XML accessibility
  tree returned by ``GET /accessibility``. Free, no LLM call. The
  desktop analog of :func:`browsergym_wrapper.heuristic.obs_to_schema`.
- :func:`osworld_wrapper.adapter.generate_label` /
  :func:`osworld_wrapper.adapter.osworld_obs_to_schema` — Head 2:
  desktop screenshot → vision LLM (GPT-4o by default) → ``<state>``
  schema.
- :func:`osworld_wrapper.grounding.grounding_osworld_obs_to_schema` —
  Head 3: desktop screenshot → OmniParser-v2 (YOLO + OCR + Florence-2)
  → ``<state>`` schema. Delegates to :mod:`browsergym_wrapper.grounding`
  with ``domain="desktop"`` since the OmniParser pipeline is
  domain-agnostic. Optional — only loaded when the heavyweight vision
  extras are installed.

**Tools** — :func:`osworld_wrapper.tools.build_osworld_registry` exposes
a :class:`vlm_wrapper.tools.ToolRegistry` of OS-level accessibility-tree
helpers (``query_os_element``, ``query_entity_pos``,
``get_state_flags``) for :mod:`vlm_wrapper.tool_loop` to use during
multi-turn grounding.

See ``osworld_wrapper/README.md`` for install instructions (Docker
image + Ubuntu qcow2 download), live-VM render quickstart, and
measured latencies.

The legacy modules ``vlm_wrapper.osworld_adapter`` and the OSWorld parts
of ``vlm_wrapper.tools_browser`` / ``vlm_wrapper.grounding_browsergym``
remain as thin re-export shims, so existing imports keep working.
"""

from __future__ import annotations

from osworld_wrapper.adapter import (
    generate_label,
    osworld_obs_to_schema,
)
from osworld_wrapper.heuristic import obs_to_schema, xml_to_schema
from osworld_wrapper.tools import build_osworld_registry

__all__ = [
    "generate_label",
    "osworld_obs_to_schema",
    "obs_to_schema",
    "xml_to_schema",
    "build_osworld_registry",
]

# OmniParser-backed grounding pulls in the heavyweight ``vlm_wrapper.grounding``
# (torch / transformers / OmniParser-v2). Make it optional so a thin install
# without the vision extras can still use Head 1 and Head 2.
try:
    from osworld_wrapper.grounding import grounding_osworld_obs_to_schema

    __all__ += ["grounding_osworld_obs_to_schema"]
except ImportError:
    pass
