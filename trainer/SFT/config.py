"""Configuration for SFT cold-start training.

LoRA hyper-parameters and target modules are kept identical to
``trainer.coevolution.config`` so the resulting adapters can be loaded
directly by the GRPO / FSDP trainer without any conversion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DECISION_ADAPTERS = ["skill_selection", "action_taking"]
SKILLBANK_ADAPTERS = ["segment", "contract", "curator"]
ALL_ADAPTERS = DECISION_ADAPTERS + SKILLBANK_ADAPTERS

DECISION_DATA_DIR = REPO_ROOT / "labeling" / "output" / "gpt54_skill_labeled" / "grpo_coldstart"
SKILLBANK_DATA_DIR = REPO_ROOT / "skill_agents_grpo" / "extract_skillbank" / "output" / "gpt54_skillbank_grpo"

COLDSTART_GAMES = [
    "avalon",
    "candy_crush",
    "diplomacy",
    "pokemon_red",
    "sokoban",
    "super_mario",
    "tetris",
    "twenty_forty_eight",
]

# module → adapter mapping for coldstart_io_all.jsonl
#
# The cold-start extraction didn't run Stage 3 (contract learning) or
# Stage 4 (bank maintenance/curation), so there are no exact-match
# records for the ``contract`` or ``curator`` adapters.  We map the
# closest available proxy data instead:
#
#   contract ← boundary_proposal (predicate analysis, ~1.4k examples)
#              + pipeline (predicate extraction + protocol synthesis,
#                ~200 examples).  These teach the model to analyze
#              trajectory states and produce structured JSON — the same
#              domain knowledge needed for effect summarization.
#
#   curator  ← skill_naming (skill name + description generation,
#              ~216 examples).  Shares domain overlap (evaluating
#              skills) but the task format differs from the co-evolution
#              approve/veto/defer prompt.  With only 216 examples this
#              adapter benefits most from the higher epoch count.
#
# GRPO training during co-evolution refines these approximations to
# the actual task-specific prompts.
COLDSTART_IO_MODULE_MAP: Dict[str, str] = {
    "boundary_proposal": "contract",
    "pipeline": "contract",
    "skill_naming": "curator",
}


@dataclass
class SFTConfig:
    """Configuration for SFT cold-start training of all 5 LoRA adapters."""

    # Base model — must match what co-evolution uses
    model_name: str = "Qwen/Qwen3-8B"

    # Data sources
    decision_data_dir: str = str(DECISION_DATA_DIR)
    skillbank_data_dir: str = str(SKILLBANK_DATA_DIR)
    games: List[str] = field(default_factory=lambda: list(COLDSTART_GAMES))

    # Output — adapters written to decision/ and skillbank/ subdirs
    output_dir: str = str(REPO_ROOT / "runs" / "sft_coldstart")

    # LoRA — matches trainer.coevolution.config exactly (no down_proj)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Training
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.05
    eval_fraction: float = 0.05
    bf16: bool = True

    # Logging / saving
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2

    # Which adapters to train (subset of ALL_ADAPTERS; None = all)
    adapters: Optional[List[str]] = None

    # Per-adapter overrides (adapter_name → {param: value}).
    # Defaults scale epochs inversely with dataset size so small-data
    # adapters (curator ~216, contract ~1.6k, segment ~2.7k) get enough
    # gradient steps to converge, while large decision adapters (~34k
    # each) don't over-train.
    adapter_overrides: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "skill_selection": {"epochs": 3},
            "action_taking":   {"epochs": int(__import__('os').environ.get("SFT_ACTION_TAKING_EPOCHS", "3"))},
            "segment":         {"epochs": 8, "batch_size": 2, "grad_accum": 8},
            "contract":        {"epochs": 10, "batch_size": 2, "grad_accum": 8},
            "curator":         {"epochs": 15, "lr": 1e-4},
        }
    )

    def resolve_target_modules(self) -> List[str]:
        """Return target_modules, auto-detecting for Qwen if unset."""
        if self.lora_target_modules is not None:
            return self.lora_target_modules
        from transformers import AutoConfig
        model_cfg = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        arch = getattr(model_cfg, "model_type", "")
        if "qwen" in arch.lower():
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
        return ["q_proj", "v_proj"]

    def adapter_output_path(self, name: str) -> Path:
        """Return the output directory for a given adapter.

        Layout matches co-evolution's ``config.adapter_path()``:
          ``<output_dir>/decision/<name>``  or
          ``<output_dir>/skillbank/<name>``
        """
        if name in DECISION_ADAPTERS:
            return Path(self.output_dir) / "decision" / name
        return Path(self.output_dir) / "skillbank" / name

    @property
    def adapters_to_train(self) -> List[str]:
        if self.adapters:
            return [a for a in self.adapters if a in ALL_ADAPTERS]
        return list(ALL_ADAPTERS)

    def effective_params(self, adapter_name: str) -> Dict[str, Any]:
        """Merge per-adapter overrides on top of the global defaults."""
        base = {
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_seq_length": self.max_seq_length,
        }
        overrides = self.adapter_overrides.get(adapter_name, {})
        base.update(overrides)
        return base
