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

# ── Default data sources (current dual-axis SFT corpus) ──────────────
#
# Decision adapters consume the per-game JSONLs produced by
# ``labeling/build_decision_sft_jsonl.py`` (output dir is timestamped;
# the auto-resolver below picks the most recent run).
DECISION_DATA_DIR = REPO_ROOT / "labeling" / "decision_sft_jsonl"

# Skill-bank adapters consume the per-game artefacts produced by
# ``labeling/run_skill_discovery.sh`` (skill_bank.jsonl,
# teacher_io_coldstart.jsonl, coldstart_io_all.jsonl).  The new layout
# inserts a ``<corpus>`` level (``gym_v`` / ``env_wrappers``) between
# the run dir and the game dir; ``data_loader.py`` walks it.
SKILLBANK_DATA_DIR = REPO_ROOT / "labeling" / "skill_bank_out"


def _latest_run(parent: Path, prefix: str = "run_") -> Path | None:
    """Return the most-recently-modified ``run_*`` dir under *parent*."""
    if not parent.is_dir():
        return None
    runs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


# The two SFT corpora are timestamped (``run_<ts>``).  When the user
# does not pin a specific run via ``--decision_data_dir`` /
# ``--skillbank_data_dir`` we resolve to the latest one available.
DEFAULT_DECISION_DATA_DIR = _latest_run(DECISION_DATA_DIR) or DECISION_DATA_DIR
DEFAULT_SKILLBANK_DATA_DIR = _latest_run(SKILLBANK_DATA_DIR) or SKILLBANK_DATA_DIR


# All 17 games covered by the dual-axis SFT corpus
# (13 Gym-V Temporal envs + 4 env_wrappers games).  data_loader.py
# auto-skips games missing data so this list is forward-compatible
# with future adds.
COLDSTART_GAMES = [
    # Gym-V (stable-retro / Genesis ROMs)
    "Temporal_Airstriker-v0",
    "Temporal_AlteredBeast-v0",
    "Temporal_CastleOfIllusion-v0",
    "Temporal_CastlevaniaBloodlines-v0",
    "Temporal_Columns-v0",
    "Temporal_DynamiteHeaddy-v0",
    "Temporal_GoldenAxe-v0",
    "Temporal_KidChameleon-v0",
    "Temporal_MortalKombatII-v0",
    "Temporal_SpaceHarrierII-v0",
    "Temporal_StreetsOfRage2-v0",
    "Temporal_Strider-v0",
    "Temporal_ThunderForceIII-v0",
    # env_wrappers (game-ai-agent + orak-mario)
    "candy_crush",
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
    model_name: str = "Qwen/Qwen3.5-9B"

    # Data sources — point at the latest dual-axis SFT corpus by default.
    # Both paths can be overridden on the CLI; ``data_loader.py`` walks
    # both legacy ``<root>/<game>/`` and the new ``<root>/<corpus>/<game>/``
    # layout so old corpora remain consumable.
    decision_data_dir: str = str(DEFAULT_DECISION_DATA_DIR)
    skillbank_data_dir: str = str(DEFAULT_SKILLBANK_DATA_DIR)
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

    # ── Lever B: scale effective batch + LR together (T2.12) ──────────
    # ``scale_effective_batch`` multiplies every per-adapter
    # ``batch_size`` (cap-aware).  Use 2.0 to go from effective-16 to
    # effective-32, etc.  ``scale_lr`` does the matching LR scaling so
    # callers can apply the linear-scale rule with a single pair of
    # flags.  Defaults to 1.0 (no change) so the cold-start baseline
    # stays identical.
    scale_effective_batch: float = 1.0
    scale_lr: float = 1.0

    # ── Speed / kernel knobs (T2.11 closure) ────────────────────────────
    # When True (default) we apply ``liger-kernel``'s fused Qwen3.5
    # patches before instantiating the base model.  Worth ~30-40 %
    # throughput on H200 / bf16.  Set to False to bisect a regression.
    use_liger_kernel: bool = True
    # Activation checkpointing trades compute for memory.  Defaults to
    # **True** for safety — at bs=16 + seq=2048 + Qwen3.5-9B with no
    # checkpointing, activations alone push ~80 GB and the model OOMs
    # on a single H200 (137 GB allocated → only ~190 MB free).  To take
    # the Lever C ~30-40 % throughput win, run multi-GPU with
    # ``--gpus_per_adapter 2+`` (or drop ``per_device_batch_size``)
    # AND pass ``--no_gradient_checkpointing`` explicitly.
    gradient_checkpointing: bool = True
    # ``None`` → :func:`speed_utils.pick_optim`.  Set explicitly to
    # 'paged_adamw_8bit' / 'adamw_torch_fused' / 'adamw_torch' to override.
    optim: Optional[str] = None
    dataloader_num_workers: int = 4
    # When True, abort training if any architecturally-required
    # projection has zero LoRA-wrapped layers (catches T2.11 drift
    # before we burn another full SFT run).
    strict_lora_coverage: bool = False

    # Per-adapter overrides (adapter_name → {param: value}).
    #
    # **Effective batch size held constant at 16** across every
    # decision / skill-bank adapter so loss curves stay comparable to
    # earlier runs.  Within that constraint, every adapter that
    # previously used grad-accumulation now collapses it into a single
    # ``batch_size`` micro-batch — H200 has 143 GB and the 9 B base +
    # paged_adamw_8bit + LoRA-only gradients fit ``bs=16`` comfortably
    # under any sequence length we're shipping.  See T2.12 in
    # ``implementation_notes/pre-training-readiness-audit.md`` for the
    # memory math.  Throughput uplift is 10-30 % from kernel-launch
    # amortisation + ``group_by_length`` pad savings, on top of the
    # liger-kernel + paged-AdamW gains.
    #
    # To push beyond effective-16 (Lever B in T2.12) the caller should
    # *also* re-tune the LR (linear-scale rule of thumb).
    adapter_overrides: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            # Big decision LoRAs (~31 k samples): 2 epochs is still a
            # standard SFT budget for cold-start; bs=16 ga=1 saturates
            # the H200 BF16 path with the 9 B base model.
            "skill_selection": {"epochs": 2, "batch_size": 16, "grad_accum": 1},
            "action_taking":   {"epochs": 2, "batch_size": 16, "grad_accum": 1},
            # Segment has the longest sequences in the corpus; with
            # ``group_by_length=True`` and paged_adamw_8bit, bs=16 fits
            # well under 100 GB even with grad-checkpointing off.
            # Collapsed ga 4→1 for ~25 % wall-clock vs the previous
            # bs=4 ga=4 schedule (same effective batch, same loss curve).
            "segment":         {"epochs": 4, "batch_size": 16, "grad_accum": 1},
            # Contract / boundary-proposal data is short — bs=16 ga=1
            # is the obvious win there (was bs=8 ga=2).
            "contract":        {"epochs": 5, "batch_size": 16, "grad_accum": 1},
            # curator: 216-sample corpus × 15 epochs is dominated by
            # per-step overhead.  bs=16 ga=1 (was bs=4 ga=4) cuts the
            # number of optimizer steps 4× — biggest wall-clock win
            # of the suite.
            "curator":         {"epochs": 15, "batch_size": 16, "grad_accum": 1, "lr": 1e-4},
        }
    )

    def resolve_target_modules(self) -> List[str]:
        """Return target_modules, auto-detecting for Qwen if unset.

        Delegates to :mod:`trainer.SFT.lora_targets` so the Qwen3.5
        hybrid-stack list (full GatedDeltaNet legs incl. ``in_proj_z/b/a``)
        is always in sync between the cold-start adapters and the
        ``schema_gen`` adapter.  See ``T2.11`` in
        ``implementation_notes/pre-training-readiness-audit.md``.
        """
        from trainer.SFT.lora_targets import resolve_target_modules as _resolve

        return _resolve(
            model_name_or_arch=self.model_name,
            explicit=self.lora_target_modules,
        )

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
        """Merge per-adapter overrides on top of the global defaults.

        Applies ``scale_effective_batch`` / ``scale_lr`` last so a
        single CLI pair (``--scale_effective_batch 2.0 --scale_lr 2.0``)
        cleanly bumps every adapter from effective-16 to effective-32
        with the linear-LR-scale rule baked in.
        """
        base = {
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_seq_length": self.max_seq_length,
        }
        overrides = self.adapter_overrides.get(adapter_name, {})
        base.update(overrides)
        # Apply the global scale factors last (Lever B).
        if self.scale_effective_batch != 1.0:
            base["batch_size"] = max(1, int(round(base["batch_size"] * self.scale_effective_batch)))
        if self.scale_lr != 1.0:
            base["lr"] = float(base["lr"]) * float(self.scale_lr)
        return base
