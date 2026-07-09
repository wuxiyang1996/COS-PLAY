#!/usr/bin/env python
"""
DEPRECATED — use ``generate_gymv_image_schema.py`` instead.

This module used to be the gymv-only vision rollout driver, calling
``gymv_wrapper.adapter.generate_label`` per frame. It has been retired in
favour of the cross-domain visual grounding head implemented in
``generate_gymv_image_schema.py``, which is the unified entry point for
all visual grounding tasks (gymv, env wrappers, benchmark image-QA, etc.).

What this shim does:
  * Emits a one-time ``DeprecationWarning`` (visible by default on stderr).
  * Forwards ``sys.argv`` to ``generate_gymv_image_schema.main()`` unchanged.

Both scripts accept the same CLI flags (``--envs``, ``--episodes``,
``--max_steps``, ``--model``, ``--temperature``, ``--max_tokens``,
``--api_key``, ``--base_url``, ``--dry_run``, ``--output_dir``, ``-v``)
so existing invocations keep working. Output now lands under
``visual_grounding_tests/output/gymv_image/...`` (was ``gpt55_gymv``)
and per-step records use the unified record schema (``schema_image_llm``,
``image_path``, ``head: "image"``) — update any downstream readers
accordingly.

Migration:
    OLD:  python visual_grounding_tests/generate_gymv_visual_schema.py ...
    NEW:  python visual_grounding_tests/generate_gymv_image_schema.py ...
"""

from __future__ import annotations

import sys
import warnings

from generate_gymv_image_schema import main as _unified_main

_DEPRECATION_MESSAGE = (
    "generate_gymv_visual_schema.py is deprecated and will be removed in a "
    "future release. Use generate_gymv_image_schema.py — the unified "
    "cross-domain visual grounding head — instead. CLI flags are unchanged; "
    "outputs now land under output/gymv_image/ and per-step records use "
    "the unified schema (schema_image_llm, image_path, head=\"image\")."
)


def main() -> None:
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    print(f"[DEPRECATED] {_DEPRECATION_MESSAGE}", file=sys.stderr)
    _unified_main()


if __name__ == "__main__":
    main()
