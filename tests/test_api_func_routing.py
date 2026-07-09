"""Tests for ``API_func`` per-model vLLM URL routing.

Pins the contract for ``VLLM_BASE_URL_MAP`` introduced when wiring the
35B-A3B control-plane backbone alongside the 9B actor: a single
``ask_model(...)`` call must dispatch to the right endpoint based on
the ``model=`` argument while preserving prior single-endpoint and
round-robin behaviour for unmapped models.

Without this test, a future refactor of ``_init_vllm_urls`` /
``_candidate_vllm_urls`` could silently regress the
``Qwen/Qwen3.5-35B-A3B`` calls back to the 9B-actor port and either
fail with a model-mismatch error from vLLM or — worse — silently
return 9B completions, defeating the point of having a separate
control-plane teacher (per ``common/models.py``
``BACKBONE_TEACHER_MODEL``).
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def fresh_api_func(monkeypatch):
    """Reload ``API_func`` with a clean vLLM-URL env. Each test gets to
    set its own ``VLLM_BASE_URL`` / ``VLLM_BASE_URLS`` /
    ``VLLM_BASE_URL_MAP`` and observes the parsed module state."""
    for var in ("VLLM_BASE_URL", "VLLM_BASE_URLS", "VLLM_BASE_URL_MAP"):
        monkeypatch.delenv(var, raising=False)

    def _load(**env: str):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        import API_func
        importlib.reload(API_func)
        API_func._init_vllm_urls()
        return API_func

    return _load


class TestParseUrlMap:
    def test_empty_map_yields_empty_dict(self, fresh_api_func) -> None:
        api = fresh_api_func()
        assert api._parse_url_map("") == {}

    def test_single_entry(self, fresh_api_func) -> None:
        api = fresh_api_func()
        assert api._parse_url_map(
            "Qwen/Qwen3.5-9B=http://localhost:8000/v1",
        ) == {"Qwen/Qwen3.5-9B": "http://localhost:8000/v1"}

    def test_multiple_entries_with_whitespace(self, fresh_api_func) -> None:
        api = fresh_api_func()
        raw = (
            "Qwen/Qwen3.5-9B=http://localhost:8000/v1, "
            "Qwen/Qwen3.5-35B-A3B = http://localhost:8001/v1"
        )
        assert api._parse_url_map(raw) == {
            "Qwen/Qwen3.5-9B": "http://localhost:8000/v1",
            "Qwen/Qwen3.5-35B-A3B": "http://localhost:8001/v1",
        }

    def test_malformed_entries_silently_dropped(self, fresh_api_func) -> None:
        api = fresh_api_func()
        raw = (
            "no_equals_sign,"
            "=missing_model_id,"
            "good=http://ok/v1,"
            ",,,"
        )
        assert api._parse_url_map(raw) == {"good": "http://ok/v1"}


class TestCandidateUrlSelection:
    def test_no_map_falls_back_to_single_url(self, fresh_api_func) -> None:
        api = fresh_api_func(VLLM_BASE_URL="http://localhost:8000/v1")
        assert api._candidate_vllm_urls(None) == ["http://localhost:8000/v1"]
        assert api._candidate_vllm_urls("Qwen/Qwen3.5-9B") == [
            "http://localhost:8000/v1",
        ]

    def test_no_map_falls_back_to_round_robin_pool(self, fresh_api_func) -> None:
        api = fresh_api_func(
            VLLM_BASE_URLS="http://a/v1,http://b/v1,http://c/v1",
        )
        candidates = api._candidate_vllm_urls("Qwen/Qwen3.5-9B")
        assert set(candidates) == {"http://a/v1", "http://b/v1", "http://c/v1"}

    def test_mapped_model_lands_on_mapped_url_first(
        self, fresh_api_func,
    ) -> None:
        api = fresh_api_func(
            VLLM_BASE_URL="http://localhost:8000/v1",
            VLLM_BASE_URL_MAP=(
                "Qwen/Qwen3.5-35B-A3B=http://localhost:8001/v1,"
                "Qwen/Qwen3.5-9B=http://localhost:8000/v1"
            ),
        )
        assert api._candidate_vllm_urls("Qwen/Qwen3.5-35B-A3B")[0] == (
            "http://localhost:8001/v1"
        )
        assert api._candidate_vllm_urls("Qwen/Qwen3.5-9B")[0] == (
            "http://localhost:8000/v1"
        )

    def test_unmapped_model_uses_round_robin_pool(self, fresh_api_func) -> None:
        api = fresh_api_func(
            VLLM_BASE_URL="http://default/v1",
            VLLM_BASE_URL_MAP="Qwen/Qwen3.5-35B-A3B=http://localhost:8001/v1",
        )
        # An unmapped model falls back to the pool; the mapped URL must
        # not pollute it.
        assert api._candidate_vllm_urls("Qwen/Qwen3.5-72B-deferred") == [
            "http://default/v1",
        ]

    def test_mapped_url_falls_back_to_pool_on_failure(
        self, fresh_api_func,
    ) -> None:
        """If the mapped 35B endpoint is dead, ``ask_vllm`` should try the
        round-robin pool next so a single-instance outage doesn't silently
        kill all judge calls. This is enforced by ``_candidate_vllm_urls``
        listing the mapped URL first then appending the pool."""
        api = fresh_api_func(
            VLLM_BASE_URL="http://pool0/v1",
            VLLM_BASE_URLS="http://pool0/v1,http://pool1/v1",
            VLLM_BASE_URL_MAP="Qwen/Qwen3.5-35B-A3B=http://mapped/v1",
        )
        candidates = api._candidate_vllm_urls("Qwen/Qwen3.5-35B-A3B")
        assert candidates[0] == "http://mapped/v1"
        assert "http://pool0/v1" in candidates
        assert "http://pool1/v1" in candidates
        # No duplicates if a pool URL also appears as the mapped URL.
        api2 = fresh_api_func(
            VLLM_BASE_URLS="http://both/v1",
            VLLM_BASE_URL_MAP="Qwen/Qwen3.5-35B-A3B=http://both/v1",
        )
        assert api2._candidate_vllm_urls("Qwen/Qwen3.5-35B-A3B") == [
            "http://both/v1",
        ]
