"""Shared pytest fixtures for ``vlm_wrapper/tests``.

The live tests here look for ``OPENAI_API_KEY`` / ``VLM_TEST_API_KEY``
in the environment.  When developers run ``pytest`` from a plain shell
those vars aren't set — but they ARE in the project's ``.env`` (which
``scripts/run_vlm_parser.py`` and ``scripts/test_vlm_parsers.py``
already auto-source).  To keep the two paths consistent we apply the
same minimal loader here, without adding a ``python-dotenv``
dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path | None = None) -> None:
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()
