"""
Lightweight tests for the multi-LoRA skill-bank agent infrastructure.

These tests verify adapter dispatch logic, config parsing, and fallback
behavior WITHOUT loading a real model (no GPU required).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from skill_agents.lora.skill_function import SkillFunction
from skill_agents.lora.config import MultiLoraConfig, LoraTrainingConfig
from skill_agents.lora.model import MultiLoraSkillBankLLM


# ── SkillFunction enum tests ────────────────────────────────────────

class TestSkillFunction:
    def test_all_values(self):
        assert SkillFunction.BOUNDARY.value == "boundary"
        assert SkillFunction.SEGMENT.value == "segment"
        assert SkillFunction.CONTRACT.value == "contract"
        assert SkillFunction.RETRIEVAL.value == "retrieval"

    def test_from_str_case_insensitive(self):
        assert SkillFunction.from_str("Boundary") == SkillFunction.BOUNDARY
        assert SkillFunction.from_str("SEGMENT") == SkillFunction.SEGMENT
        assert SkillFunction.from_str("contract") == SkillFunction.CONTRACT
        assert SkillFunction.from_str("Retrieval") == SkillFunction.RETRIEVAL

    def test_from_str_invalid(self):
        with pytest.raises(ValueError, match="Unknown skill function"):
            SkillFunction.from_str("invalid")

    def test_adapter_name(self):
        for fn in SkillFunction:
            assert fn.adapter_name == fn.value

    def test_iteration(self):
        fns = list(SkillFunction)
        assert len(fns) == 4


# ── MultiLoraConfig tests ───────────────────────────────────────────

class TestMultiLoraConfig:
    def test_defaults(self):
        cfg = MultiLoraConfig()
        assert cfg.base_model_name_or_path == "Qwen/Qwen3-8B"
        assert cfg.adapter_paths == {}
        assert cfg.allow_fallback_to_base_model is True

    def test_adapter_path_for(self):
        cfg = MultiLoraConfig(adapter_paths={"boundary": "/tmp/b", "segment": "/tmp/s"})
        assert cfg.adapter_path_for(SkillFunction.BOUNDARY) == "/tmp/b"
        assert cfg.adapter_path_for(SkillFunction.SEGMENT) == "/tmp/s"
        assert cfg.adapter_path_for(SkillFunction.CONTRACT) is None

    def test_has_adapter_missing_path(self):
        cfg = MultiLoraConfig(adapter_paths={"boundary": "/nonexistent/path"})
        assert cfg.has_adapter(SkillFunction.BOUNDARY) is False

    def test_has_adapter_existing_path(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = MultiLoraConfig(adapter_paths={"boundary": d})
            assert cfg.has_adapter(SkillFunction.BOUNDARY) is True

    def test_from_dict(self):
        d = {
            "base_model_name_or_path": "test/model",
            "adapter_paths": {"boundary": "/a", "segment": "/b"},
            "temperature": 0.5,
            "extra_key": "ignored",
        }
        cfg = MultiLoraConfig.from_dict(d)
        assert cfg.base_model_name_or_path == "test/model"
        assert cfg.temperature == 0.5
        assert len(cfg.adapter_paths) == 2

    def test_to_dict(self):
        cfg = MultiLoraConfig(adapter_paths={"boundary": "/a"})
        d = cfg.to_dict()
        assert d["adapter_paths"] == {"boundary": "/a"}
        assert "base_model_name_or_path" in d


# ── LoraTrainingConfig tests ────────────────────────────────────────

class TestLoraTrainingConfig:
    def test_function_enum(self):
        cfg = LoraTrainingConfig(skill_function="segment")
        assert cfg.function_enum == SkillFunction.SEGMENT

    def test_defaults(self):
        cfg = LoraTrainingConfig()
        assert cfg.lora_r == 16
        assert cfg.num_train_epochs == 3


# ── MultiLoraSkillBankLLM dispatch tests ────────────────────────────

class TestMultiLoraSkillBankLLM:
    def test_initial_state(self):
        cfg = MultiLoraConfig()
        llm = MultiLoraSkillBankLLM(cfg)
        assert not llm.is_loaded
        assert llm.loaded_adapters == []
        assert llm.active_adapter is None

    def test_status(self):
        cfg = MultiLoraConfig(adapter_paths={"boundary": "/tmp/b"})
        llm = MultiLoraSkillBankLLM(cfg)
        s = llm.status()
        assert s["base_model"] == "Qwen/Qwen3-8B"
        assert not s["is_loaded"]
        assert s["configured_adapters"]["boundary"] == "/tmp/b"
        assert s["configured_adapters"]["contract"] is None

    def test_shared_instance(self):
        cfg = MultiLoraConfig()
        llm = MultiLoraSkillBankLLM(cfg)
        assert MultiLoraSkillBankLLM.get_shared_instance() is None

        MultiLoraSkillBankLLM.set_shared_instance(llm)
        assert MultiLoraSkillBankLLM.get_shared_instance() is llm

        MultiLoraSkillBankLLM.set_shared_instance(None)
        assert MultiLoraSkillBankLLM.get_shared_instance() is None

    def test_as_ask_fn_returns_callable(self):
        cfg = MultiLoraConfig()
        llm = MultiLoraSkillBankLLM(cfg)
        fn = llm.as_ask_fn(SkillFunction.BOUNDARY)
        assert callable(fn)

    @patch.object(MultiLoraSkillBankLLM, "generate", return_value="mocked output")
    @patch.object(MultiLoraSkillBankLLM, "load")
    def test_as_ask_fn_routes_correctly(self, mock_load, mock_gen):
        cfg = MultiLoraConfig()
        llm = MultiLoraSkillBankLLM(cfg)
        llm._model = True  # pretend loaded

        ask = llm.as_ask_fn(SkillFunction.BOUNDARY)
        result = ask("test prompt", model="ignored", temperature=0.2, max_tokens=500)

        assert result == "mocked output"
        mock_gen.assert_called_once()
        call_args = mock_gen.call_args
        assert call_args[0] == (SkillFunction.BOUNDARY, "test prompt")
        assert call_args[1]["temperature"] == 0.2
        assert call_args[1]["max_new_tokens"] == 500

    def test_function_to_adapter_mapping(self):
        """Verify every SkillFunction maps to a unique adapter name."""
        names = {fn.adapter_name for fn in SkillFunction}
        assert len(names) == 4
        assert names == {"boundary", "segment", "contract", "retrieval"}


# ── Fallback behavior tests ─────────────────────────────────────────

class TestFallbackBehavior:
    def test_no_lora_config_returns_none_shared_instance(self):
        """Without set_shared_instance, get_shared_instance returns None."""
        MultiLoraSkillBankLLM.set_shared_instance(None)
        assert MultiLoraSkillBankLLM.get_shared_instance() is None

    def test_config_without_adapters_allows_fallback(self):
        cfg = MultiLoraConfig(
            adapter_paths={},
            allow_fallback_to_base_model=True,
        )
        llm = MultiLoraSkillBankLLM(cfg)
        assert llm.config.allow_fallback_to_base_model

    def test_config_no_fallback_raises(self):
        cfg = MultiLoraConfig(
            adapter_paths={},
            allow_fallback_to_base_model=False,
        )
        llm = MultiLoraSkillBankLLM(cfg)
        llm._model = MagicMock()
        llm._tokenizer = MagicMock()
        llm._is_peft_model = False

        with pytest.raises(RuntimeError, match="allow_fallback_to_base_model=False"):
            llm._activate_adapter(SkillFunction.BOUNDARY)


# ── Data builder import test ─────────────────────────────────────────

class TestDataBuilders:
    def test_import(self):
        from trainer.skillbank.lora.data_builder import (
            build_boundary_dataset,
            build_segment_dataset,
            build_contract_dataset,
            build_retrieval_dataset,
            BUILDERS,
        )
        assert SkillFunction.BOUNDARY in BUILDERS
        assert SkillFunction.SEGMENT in BUILDERS
        assert SkillFunction.CONTRACT in BUILDERS
        assert SkillFunction.RETRIEVAL in BUILDERS

    def test_write_and_load_jsonl(self):
        from trainer.skillbank.lora.data_builder import _write_jsonl, load_jsonl_dataset

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            examples = [{"prompt": "hello", "completion": "world"}]
            _write_jsonl(examples, path)
            loaded = load_jsonl_dataset(path)
            assert len(loaded) == 1
            assert loaded[0]["prompt"] == "hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
