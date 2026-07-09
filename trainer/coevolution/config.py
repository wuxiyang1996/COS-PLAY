"""Configuration for the co-evolution training loop."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))


SKILL_BANK_GAMES = [
    "diplomacy",
    "twenty_forty_eight",
    "tetris",
    "avalon",
    "sokoban",
    "candy_crush",
]

# Evaluation-only: rollouts collected for metrics but NOT fed into GRPO training.
# pokemon_red: PyBoy emulator race condition under concurrent episode init.
# super_mario: nes_py 8.2.1 + NumPy 2.x incompatibility.
EVAL_ONLY_GAMES: List[str] = []

GAME_MAX_STEPS: Dict[str, int] = {
    "pokemon_red": 200,
    "diplomacy": 20,
    "twenty_forty_eight": 200,
    "tetris": 200,
    "avalon": 50,
    "sokoban": 200,
    "candy_crush": 50,
    "super_mario": 500,  # kept for reference if re-enabled
}

EMULATOR_GAMES = {"pokemon_red"}

# ── Multi-role rollout constants ─────────────────────────────────────
# When ``unified_role_rollouts`` is enabled, the same decision agent
# plays ALL roles and each rollout is tagged with role / side metadata.
# Per-game episode overrides ensure sufficient role coverage.

EPISODES_PER_GAME_MULTIROLE: Dict[str, int] = {
    "avalon": 5,
    "diplomacy": 7,
}

AVALON_ROLES: List[str] = ["Merlin", "Servant", "Servant", "Minion", "Assassin"]
AVALON_SIDES: Dict[str, str] = {
    "Merlin": "good", "Percival": "good", "Servant": "good",
    "Mordred": "evil", "Morgana": "evil", "Oberon": "evil",
    "Minion": "evil", "Assassin": "evil",
}
DIPLOMACY_POWERS: List[str] = [
    "AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY",
]


def resolve_bank_key(game: str, role: str = "", side: str = "") -> str:
    """Return the skill-bank routing key for an episode.

    In unified-role mode the key encodes the role dimension so each
    side (Avalon) or power (Diplomacy) gets its own bank.  For all
    other games the key is just the game name.

    Examples::

        resolve_bank_key("avalon", "Merlin", "good")   -> "avalon/good"
        resolve_bank_key("avalon", "Assassin", "evil")  -> "avalon/evil"
        resolve_bank_key("diplomacy", "FRANCE", "FRANCE") -> "diplomacy/FRANCE"
        resolve_bank_key("tetris")                      -> "tetris"
    """
    if game == "avalon" and side:
        return f"avalon/{side}"
    if game == "diplomacy" and role:
        return f"diplomacy/{role}"
    return game


def bank_keys_for_game(game: str) -> List[str]:
    """Return all possible bank keys for a game.

    Used by ``PerGameSkillBankManager`` to pre-create sub-bank pipelines.
    """
    if game == "avalon":
        return ["avalon/good", "avalon/evil"]
    if game == "diplomacy":
        return [f"diplomacy/{p}" for p in DIPLOMACY_POWERS]
    return [game]


GAME_DURATION_ORDER = [
    "diplomacy",
    "twenty_forty_eight",
    "tetris",
    "avalon",
    "sokoban",
    "candy_crush",
    "pokemon_red",
]

ADAPTER_NAMES = [
    "skill_selection",
    "action_taking",
    "segment",
    "contract",
    "curator",
]

# ── Curriculum presets ───────────────────────────────────────────────
# Each maps step thresholds → active game lists.

CURRICULUM_GRADUAL: Dict[int, List[str]] = {
    0: ["twenty_forty_eight", "tetris", "candy_crush"],
    10: ["twenty_forty_eight", "tetris", "candy_crush", "sokoban"],
    15: ["twenty_forty_eight", "tetris", "candy_crush", "sokoban", "avalon"],
    20: ["twenty_forty_eight", "tetris", "candy_crush", "sokoban", "avalon", "diplomacy"],
}

CURRICULUM_FOCUSED: Dict[int, List[str]] = {
    0: ["twenty_forty_eight", "tetris", "candy_crush", "sokoban"],
    40: ["avalon", "diplomacy"],
}

CURRICULUM_PRESETS: Dict[str, Optional[Dict[int, List[str]]]] = {
    "gradual": CURRICULUM_GRADUAL,
    "focused": CURRICULUM_FOCUSED,
    "none": None,
}


def _model_short_name(model_name: str) -> str:
    """Extract a filesystem-safe short name from a model identifier.

    ``"Qwen/Qwen3-8B"`` → ``"Qwen3-8B"``
    ``"meta-llama/Llama-3-8B"`` → ``"Llama-3-8B"``
    """
    short = model_name.rsplit("/", 1)[-1]
    return re.sub(r"[^\w\-.]", "_", short)


def _generate_run_dir(model_name: str) -> str:
    """Generate a unique run directory name from model name + timestamp.

    Example: ``runs/Qwen3-8B_20260315_143022``
    """
    short = _model_short_name(model_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("runs") / f"{short}_{ts}")


@dataclass
class CoEvolutionConfig:
    """Top-level configuration for the co-evolution loop."""

    games: List[str] = field(default_factory=lambda: list(SKILL_BANK_GAMES))
    eval_games: List[str] = field(default_factory=lambda: list(EVAL_ONLY_GAMES))
    episodes_per_game: int = 4
    eval_episodes_per_game: int = 3

    # ── Unified multi-role rollout mode ──────────────────────────
    # When True, Avalon and Diplomacy use a single shared decision
    # agent for all roles.  Rollouts are tagged with role / side /
    # stage metadata so the skill bank can segment skills along
    # those dimensions.  Episode counts follow per-game overrides
    # (default: 5 for Avalon, 7 for Diplomacy).  Other games are
    # unaffected.
    # When False (default), the legacy random-role behaviour is used
    # and ``episodes_per_game`` applies uniformly to every game.
    unified_role_rollouts: bool = False
    episodes_per_game_overrides: Dict[str, int] = field(
        default_factory=lambda: dict(EPISODES_PER_GAME_MULTIROLE),
    )

    max_concurrent_episodes: int = 64
    total_steps: int = 60

    # GPU allocation — split between persistent vLLM and FSDP training.
    # vLLM instances (TP=1) run on vllm_gpu_ids and stay up for the
    # entire training run.  After each GRPO step, adapters are hot-reloaded
    # via the vLLM API (no restart required).
    vllm_gpu_ids: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3],
    )
    grpo_devices: List[int] = field(
        default_factory=lambda: [4, 5, 6, 7],
    )

    # vLLM inference
    model_name: str = "Qwen/Qwen3-8B"
    temperature: float = 0.5
    max_tokens: int = 512
    vllm_base_url: str = "http://localhost:8000/v1"  # used only when manage_vllm=False
    vllm_base_port: int = 8000
    vllm_gpu_util: float = 0.95

    # Speculative decoding — use a small draft model to propose tokens
    # that the main model verifies in parallel (~2-3x generation speedup).
    speculative_model: Optional[str] = "Qwen/Qwen3-0.6B"
    num_speculative_tokens: int = 5

    # When True, the orchestrator manages vLLM server lifecycle
    # (persistent instances on vllm_gpu_ids, hot-reload after GRPO).
    manage_vllm: bool = True

    # External opponent for multi-agent games (Avalon, Diplomacy).
    # When set, non-controlled players/powers use this API model
    # instead of self-play via the local vLLM model.  This breaks the
    # symmetric-weakness problem where both sides share the same bugs.
    # Example: "gpt-5-mini" (routed through OpenRouter).
    opponent_model: Optional[str] = None
    opponent_api_base: Optional[str] = "https://openrouter.ai/api/v1"

    # Skill bank EM
    em_max_iterations: int = 3
    em_micro_batch_size: int = 8

    # GRPO
    grpo_enabled: bool = True
    # Deprecated: kept for backward compat only.
    grpo_decision_devices: List[int] = field(default_factory=list)
    grpo_skillbank_devices: List[int] = field(default_factory=list)

    # Run directory — all other dirs are relative to this.
    # Auto-generated from model_name + timestamp if None.
    run_dir: Optional[str] = None

    # Directories (rebased under run_dir by resolve_paths())
    bank_dir: str = "skillbank"
    adapter_dir: str = "lora_adapters"  # parent; decision/ and skillbank/ live under this
    checkpoint_dir: str = "checkpoints"
    log_dir: str = ""  # root of run_dir
    grpo_data_dir: str = "grpo_data"
    rewards_dir: str = "rewards"
    tensorboard_dir: str = "tensorboard"
    debug_io_dir: str = "debug_io"

    # Debug: log every LLM I/O and GRPO sample to disk for inspection
    debug_io: bool = False

    # Checkpointing
    checkpoint_interval: int = 5
    # How many per-step checkpoints to keep on disk.  Set to 0 to keep ALL.
    # The "best" checkpoint (step_99999) is never deleted regardless.
    checkpoint_keep_last: int = 0

    # Number of consecutive reward declines before rolling back to the
    # best checkpoint.  Higher values give the optimizer more room to
    # recover from temporary dips; lower values are more conservative.
    rollback_patience: int = 4

    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "game-ai-coevolution"
    wandb_run_name: Optional[str] = None

    # Start mode:
    #   "from_scratch" — random-init all LoRA adapters, ignore any checkpoint
    #   "resume"       — resume from latest (or specific) checkpoint
    #   "auto"         — resume if checkpoint exists, else from scratch
    start_mode: str = "auto"
    resume_from_step: Optional[int] = None

    # Load pre-trained adapters instead of random init.
    # Maps adapter name → path to an existing adapter directory.
    # Only used when start_mode != "resume" (resume loads from checkpoint).
    # Example: {"skill_selection": "prev_run/lora/skill_selection", ...}
    pretrained_adapter_paths: Dict[str, str] = field(default_factory=dict)

    # Seed each per-game skill bank from a cold-start directory on first
    # launch.  Expected layout: ``<seed_bank_dir>/<game>/skill_bank.jsonl``.
    # Skills are copied only when the game's bank is empty; once the
    # co-evolution loop adds its own skills, the seed is never re-applied.
    seed_bank_dir: Optional[str] = None

    # Thread/process executors
    thread_workers: int = 64
    process_workers: int = 8

    # Early episode termination
    stuck_window: int = 15
    min_steps_before_stuck_check: int = 20

    # Rollout batching synchronizer — prevents episodes from
    # desynchronizing and losing vLLM request batching (which causes
    # 10-20x throughput loss due to the GPU batch-size cliff).
    rollout_sync_timeout_s: float = 0.10

    # LoRA adapter defaults (matches skill_agents_grpo.lora.config)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    # "gaussian" → both A and B get small random init (better for GRPO)
    # True       → Kaiming A + zero B (standard LoRA, B gets no grad initially)
    lora_init_weights: Any = "gaussian"

    # Curriculum learning — two-phase focused training.
    # Phase 1 (steps 0-39): 4 simpler single-player games to build core skills.
    # Phase 2 (steps 40-59): focus on harder social/strategic games (Avalon, Diplomacy).
    # Set to None for no curriculum (all games from step 0).
    # Use CURRICULUM_PRESETS["gradual"] for the old incremental schedule.
    curriculum_schedule: Optional[Dict[int, List[str]]] = field(
        default_factory=lambda: dict(CURRICULUM_FOCUSED)
    )

    # GRPO from-scratch schedule — higher exploration early, then anneal.
    # Only applied when start_mode == "from_scratch" (otherwise GRPO
    # configs use the default values from StageGRPOConfig).
    scratch_warmup_steps: int = 20
    scratch_initial_lr: float = 1e-4
    scratch_steady_lr: float = 5e-5
    scratch_initial_temperature: float = 1.0
    scratch_steady_temperature: float = 0.7
    scratch_initial_kl_coeff: float = 0.01
    scratch_steady_kl_coeff: float = 0.05

    # Per-run GRPO overrides (set via CLI, leave None to use defaults)
    grpo_clip_ratio: float = 0.2
    grpo_max_epochs: int = 4
    grpo_adv_clip: Optional[float] = None

    # Ablation: disable the skill bank and skill bank agent entirely.
    # When True, rollouts run in cold-start mode (no skill retrieval),
    # Phase B (skill discovery / bank update) is skipped, and GRPO
    # trains the ``action_taking`` LoRA only — neither ``skill_selection``
    # nor the three skill-bank adapters (segment/contract/curator)
    # are updated.  Corresponds to the "GRPO w/o skill" row in the
    # paper's Table 1 ablation.
    disable_skillbank: bool = False

    # Ablation: freeze the skill bank but keep it ACTIVE during rollouts.
    # When True, the skill bank loaded at startup is used for retrieval /
    # CALL_SKILL during rollouts (so the policy can read existing skills),
    # but Phase B (skill discovery / bank update) is SKIPPED and the
    # three skill-bank adapters (segment/contract/curator) stay frozen.
    # GRPO trains the full decision agent (``action_taking`` AND
    # ``skill_selection``).  Corresponds to the
    # "GRPO w/ fixed skill bank" row in the paper's Table 1 ablation.
    # Mutually exclusive with ``disable_skillbank``.
    freeze_skillbank: bool = False

    _resolved: bool = field(default=False, repr=False)

    def resolve_paths(self) -> "CoEvolutionConfig":
        """Rebase all directory paths under ``run_dir``.

        If ``run_dir`` is ``None``, generates one from the model name
        and current timestamp (e.g. ``runs/Qwen3-8B_20260315_143022``).

        Idempotent — calling twice is safe.
        """
        if self._resolved:
            return self

        if self.run_dir is None:
            self.run_dir = _generate_run_dir(self.model_name)

        root = Path(self.run_dir).resolve()
        self.run_dir = str(root)

        def _rebase(rel: str) -> str:
            p = Path(rel)
            if p.is_absolute():
                return rel
            return str(root / rel) if rel else str(root)

        self.bank_dir = _rebase(self.bank_dir)
        self.adapter_dir = _rebase(self.adapter_dir)
        self.checkpoint_dir = _rebase(self.checkpoint_dir)
        self.log_dir = _rebase(self.log_dir) if self.log_dir else str(root)
        self.grpo_data_dir = _rebase(self.grpo_data_dir)
        self.rewards_dir = _rebase(self.rewards_dir)
        self.tensorboard_dir = _rebase(self.tensorboard_dir)
        self.debug_io_dir = _rebase(self.debug_io_dir)

        self._resolved = True
        return self

    @property
    def effective_grpo_devices(self) -> List[int]:
        """GPU IDs used for GRPO training (FSDP data-parallel)."""
        return self.grpo_devices

    @property
    def vllm_base_urls(self) -> List[str]:
        """vLLM base URLs for the inference client.

        Returns one URL per vLLM GPU when managed, or a single URL
        when the user runs vLLM externally.
        """
        if self.manage_vllm:
            return [
                f"http://localhost:{self.vllm_base_port + i}/v1"
                for i in range(len(self.vllm_gpu_ids))
            ]
        return [self.vllm_base_url]

    @property
    def decision_adapter_dir(self) -> str:
        return str(Path(self.adapter_dir) / "decision")

    @property
    def skillbank_adapter_dir(self) -> str:
        return str(Path(self.adapter_dir) / "skillbank")

    def adapter_path(self, name: str) -> str:
        if name in ("skill_selection", "action_taking"):
            return str(Path(self.decision_adapter_dir) / name)
        return str(Path(self.skillbank_adapter_dir) / name)

    def get_episodes_for_game(self, game: str) -> int:
        """Return the episode count for *game*.

        In unified-role mode, per-game overrides are applied
        (5 for Avalon, 7 for Diplomacy by default).  Otherwise
        the global ``episodes_per_game`` is returned for all games.
        """
        if self.unified_role_rollouts:
            return self.episodes_per_game_overrides.get(
                game, self.episodes_per_game,
            )
        return self.episodes_per_game

    def active_games(self, step: int) -> List[str]:
        """Return the list of active games for the given training step.

        Uses ``curriculum_schedule`` when set, otherwise returns all games.
        """
        if not self.curriculum_schedule:
            return list(self.games)
        thresholds = sorted(k for k in self.curriculum_schedule if k <= step)
        if not thresholds:
            return list(self.games)
        return list(self.curriculum_schedule[thresholds[-1]])

    def curriculum_description(self) -> str:
        """Human-readable summary of the curriculum schedule."""
        if not self.curriculum_schedule:
            return "none (all games every step)"
        phases = sorted(self.curriculum_schedule.items())
        parts = []
        for i, (start, games) in enumerate(phases):
            end = phases[i + 1][0] - 1 if i + 1 < len(phases) else self.total_steps - 1
            parts.append(f"  steps {start}–{end}: {', '.join(games)}")
        return "focused curriculum\n" + "\n".join(parts)

    def grpo_schedule(self, step: int) -> Dict[str, float]:
        """Return GRPO hyperparameters for the current step.

        During from-scratch training, the first ``scratch_warmup_steps``
        use higher learning rate, higher sampling temperature (more
        exploration), and lower KL penalty (allow larger policy shifts).
        After warmup, LR follows cosine decay to a minimum of 10% of
        steady-state.  Temperature and KL hold at steady values.
        """
        import math as _math

        if self.start_mode != "from_scratch":
            total = max(1, self.total_steps)
            progress = min(1.0, step / total)
            lr_min = self.scratch_steady_lr * 0.3
            lr = lr_min + 0.5 * (self.scratch_steady_lr - lr_min) * (
                1.0 + _math.cos(_math.pi * progress)
            )
            kl = self.scratch_steady_kl_coeff
            return {
                "lr": lr,
                "temperature": self.scratch_steady_temperature,
                "kl_coeff": kl,
            }

        w = self.scratch_warmup_steps
        total = self.total_steps

        if w <= 0 or step >= w:
            warmup_alpha = 1.0
        else:
            warmup_alpha = step / w

        def _lerp(init: float, steady: float) -> float:
            return init + warmup_alpha * (steady - init)

        if step < w:
            lr = _lerp(self.scratch_initial_lr, self.scratch_steady_lr)
        else:
            decay_steps = max(1, total - w)
            progress = min(1.0, (step - w) / decay_steps)
            lr_min = self.scratch_steady_lr * 0.1
            lr = lr_min + 0.5 * (self.scratch_steady_lr - lr_min) * (
                1.0 + _math.cos(_math.pi * progress)
            )

        return {
            "lr": lr,
            "temperature": _lerp(
                self.scratch_initial_temperature,
                self.scratch_steady_temperature,
            ),
            "kl_coeff": _lerp(
                self.scratch_initial_kl_coeff,
                self.scratch_steady_kl_coeff,
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict for config.json persistence."""
        d = asdict(self)
        d.pop("_resolved", None)
        return d


DECISION_ADAPTERS = ["skill_selection", "action_taking"]
SKILLBANK_ADAPTERS = ["segment", "contract", "curator"]


def prepare_adapters(config: CoEvolutionConfig) -> Dict[str, str]:
    """Ensure every adapter directory is populated and ready for vLLM/GRPO.

    Two paths:

    **Load pre-trained** (``config.pretrained_adapter_paths`` is non-empty):
        Copy the 2 decision + 3 skill-bank adapters from the given paths
        into ``config.adapter_dir``.  Any adapter not listed in the dict
        will be random-initialised as a fallback.

    **Train from scratch** (``config.start_mode == "from_scratch"`` or
    no pre-trained paths and no existing adapters):
        Create random-initialised adapters (``init_lora_weights="gaussian"``
        by default).  Both the A and B LoRA matrices receive small random
        values so that gradients flow to all parameters from step 1.

    Returns a dict mapping adapter name → resolved directory path.
    """
    import gc
    import logging
    import shutil

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    logger = logging.getLogger(__name__)
    force = config.start_mode == "from_scratch"
    pretrained = config.pretrained_adapter_paths or {}
    result: Dict[str, str] = {}

    # ── Phase 1: copy pre-trained adapters ────────────────────────
    copied: List[str] = []
    for name in ADAPTER_NAMES:
        dst = Path(config.adapter_path(name))
        src = pretrained.get(name)
        if src is not None:
            src_path = Path(src)
            if not (src_path / "adapter_config.json").exists():
                # PEFT save_pretrained nests files under <adapter_name>/;
                # check one level deeper before giving up.
                nested = src_path / name
                if (nested / "adapter_config.json").exists():
                    logger.info(
                        "Pre-trained adapter '%s': found nested layout at %s",
                        name, nested,
                    )
                    src_path = nested
                else:
                    # Also try any single subdirectory that has the file
                    found = False
                    if src_path.is_dir():
                        for child in src_path.iterdir():
                            if child.is_dir() and (child / "adapter_config.json").exists():
                                logger.info(
                                    "Pre-trained adapter '%s': found nested layout at %s",
                                    name, child,
                                )
                                src_path = child
                                found = True
                                break
                    if not found:
                        logger.warning(
                            "Pre-trained adapter '%s' not found at %s "
                            "(checked top-level and subdirectories) — will random-init",
                            name, src,
                        )
                        continue
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src_path), str(dst))
            logger.info("Loaded pre-trained adapter '%s': %s → %s", name, src, dst)
            copied.append(name)
            result[name] = str(dst)

    # ── Phase 2: random-init any remaining missing adapters ───────
    need_init: List[str] = []
    for name in ADAPTER_NAMES:
        if name in copied:
            continue
        dst = Path(config.adapter_path(name))
        marker = dst / "adapter_config.json"
        if marker.exists() and not force:
            logger.info("LoRA adapter '%s' already exists: %s", name, dst)
            result[name] = str(dst)
        else:
            if force and dst.exists():
                logger.info("Force re-init: removing existing adapter '%s'", name)
                shutil.rmtree(dst)
            need_init.append(name)

    if not need_init:
        if copied:
            logger.info(
                "Loaded %d pre-trained adapter(s), all adapters ready", len(copied),
            )
        return result

    # ── Resolve target_modules from model architecture ────────────
    target_modules = config.lora_target_modules
    if target_modules is None:
        model_cfg = AutoConfig.from_pretrained(
            config.model_name, trust_remote_code=True,
        )
        arch = getattr(model_cfg, "model_type", "")
        if "qwen" in arch.lower():
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj",
            ]
        else:
            target_modules = ["q_proj", "v_proj"]

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=config.lora_init_weights,
    )

    init_desc = (
        "gaussian (A & B both random — GRPO-ready)"
        if config.lora_init_weights == "gaussian"
        else f"standard ({config.lora_init_weights})"
    )
    logger.info(
        "Loading base model '%s' on CPU to initialise %d adapter(s) "
        "[init=%s, r=%d, alpha=%d]: %s",
        config.model_name, len(need_init), init_desc,
        config.lora_r, config.lora_alpha, need_init,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    for name in need_init:
        out = Path(config.adapter_path(name))
        out.mkdir(parents=True, exist_ok=True)
        logger.info("Random-init LoRA adapter '%s' → %s", name, out)
        try:
            peft_model = get_peft_model(base_model, lora_cfg)
            peft_model.save_pretrained(str(out))
            result[name] = str(out)
            base_model = peft_model.unload()
        except Exception as exc:
            logger.error("Failed to create adapter '%s': %s", name, exc)

    del base_model
    gc.collect()
    # Do NOT call torch.cuda.empty_cache() here — it initializes a CUDA
    # context on GPU 0, wasting ~6 GB that the vLLM instance needs.

    logger.info(
        "Adapter summary: %d pre-trained, %d random-init, %d total ready",
        len(copied), len(need_init), len(result),
    )
    return result


# Keep the old name as an alias for backward compatibility
def init_lora_adapters(
    config: CoEvolutionConfig,
    force: bool = False,
) -> List[str]:
    """Backward-compatible wrapper around :func:`prepare_adapters`."""
    if force:
        config.start_mode = "from_scratch"
    result = prepare_adapters(config)
    return list(result.keys())
