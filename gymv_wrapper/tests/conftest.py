"""Shared pytest fixtures for ``gymv_wrapper/tests``.

Mirrors :mod:`vlm_wrapper.tests.conftest` so live tests pick up
``OPENAI_API_KEY`` / ``VLM_TEST_API_KEY`` from the project ``.env`` even
when ``pytest`` is launched from a plain shell — no
``python-dotenv`` dependency required.
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
