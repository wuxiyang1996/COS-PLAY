"""Shared pytest fixtures for decision_agents tests.

1. Stubs out ``API_func`` and ``anthropic`` / ``openai`` if they are not
   installed, so that deep imports (``data_structure.experience`` →
   ``API_func`` → ``anthropic``) succeed in a minimal environment.
2. Patches ``decision_agents.actor_agent.ask_model`` and
   ``decision_agents.agent_helper.ask_model`` to ``None`` so that the
   offline smoke tests never touch the network.  When ``ask_model`` is
   ``None`` the actor falls through to its deterministic "first valid
   action" fallback, which is exactly what the tests rely on.
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_stub(module_name: str, attrs: dict) -> None:
    if module_name in sys.modules:
        return
    mod = types.ModuleType(module_name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[module_name] = mod


# ---------------------------------------------------------------------
# Module-level: stub optional deps before they get imported transitively.
# ---------------------------------------------------------------------


def _ensure_optional_deps_stubbed() -> None:
    try:
        import anthropic  # noqa: F401
    except Exception:
        _install_stub(
            "anthropic",
            {
                "Anthropic": type(
                    "Anthropic", (), {"__init__": lambda self, *a, **k: None}
                ),
            },
        )

    try:
        import openai  # noqa: F401
    except Exception:
        class _StubClient:
            def __init__(self, *a, **k) -> None:
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **kw: None)
                )

        _install_stub("openai", {"OpenAI": _StubClient, "Client": _StubClient})

    # ``from google import genai``.  The real ``google`` package is a
    # namespace package; we synthesise a submodule only if genai isn't
    # importable in the current env.
    try:
        from google import genai  # noqa: F401
    except Exception:
        google_mod = sys.modules.get("google")
        if google_mod is None:
            google_mod = types.ModuleType("google")
            google_mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules["google"] = google_mod
        genai_mod = types.ModuleType("google.genai")
        genai_mod.Client = type(  # type: ignore[attr-defined]
            "Client", (), {"__init__": lambda self, *a, **k: None}
        )
        sys.modules["google.genai"] = genai_mod
        setattr(google_mod, "genai", genai_mod)

    # Some of our code does ``from dotenv import load_dotenv``.
    try:
        import dotenv  # noqa: F401
    except Exception:
        _install_stub("dotenv", {"load_dotenv": lambda *a, **k: False})


_ensure_optional_deps_stubbed()


# ---------------------------------------------------------------------
# Per-test: disable live LLM calls.
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_ask_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import decision_agents.actor_agent as actor_module
    monkeypatch.setattr(actor_module, "ask_model", None, raising=False)
    try:
        import decision_agents.agent_helper as helper_module
        monkeypatch.setattr(helper_module, "ask_model", None, raising=False)
    except Exception:
        pass
