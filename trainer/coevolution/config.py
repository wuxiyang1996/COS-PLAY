"""Configuration for the co-evolution training loop."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))


SKILL_BANK_GAMES = [
    # Full 12-game registry — all P1 source + P2 hold-out games. Mirrors
    # ``GAME_MAX_STEPS`` below so ``--games`` validation, the no-args
    # default, and the curriculum scripts agree on the same universe.
    #
    # Phase split (training_notes/coevo-3phase-cross-game-ood-transfer-plan.md
    # §4.1, refreshed 2026-05-03 PM) is enforced by the curriculum
    # scripts' ``PHASES`` arrays, NOT by this constant — so:
    #   * ``scripts/run_phase1_curriculum.sh`` uses the 6 P1 picks
    #     below (4 healthy gymv + ``candy_crush`` + ``tetris``).
    #   * ``scripts/run_phase2_holdout.sh`` uses the 6 hold-outs
    #     (4 P2 gymv + ``twenty_forty_eight`` + ``super_mario``).
    #
    # ``python scripts/run_coevolution.py`` (no ``--games``) now defaults
    # to all 12. For smoke tests against a single game, prefer an
    # explicit ``--games tetris`` rather than relying on this default.
    #
    # ── Phase-1 source roster (§4.1) ──────────────────────────────
    "tetris",                    # paper Table 3
    "candy_crush",               # paper Table 3
    "gymv_thunder_force_iii",    # gymv (data-driven pick)
    "gymv_altered_beast",        # gymv (data-driven pick)
    "gymv_columns",              # gymv (data-driven pick)
    "gymv_dynamite_headdy",      # gymv (data-driven pick)
    # ── Phase-2 hold-out roster (§7.1) ────────────────────────────
    "twenty_forty_eight",        # paper Table 3
    "super_mario",               # paper Table 3
    "gymv_streets_of_rage_2",    # gymv (in-genre lift target)
    "gymv_space_harrier_ii",     # gymv (scale-jump test)
    "gymv_airstriker",           # gymv (easier in-genre sanity)
    "gymv_strider",              # gymv (partial-signal rescue test)
]

# Phase-1 source roster (§4.1) — used by curriculum scripts and
# smoke-tests that want only the healthy training subset. Defined as a
# tuple to discourage in-place mutation; convert to ``list(...)`` if you
# need to pass it to ``CoEvolutionConfig.games``.
PHASE1_DEFAULT_GAMES: tuple = (
    "tetris",
    "candy_crush",
    "gymv_thunder_force_iii",
    "gymv_altered_beast",
    "gymv_columns",
    "gymv_dynamite_headdy",
)

# Phase-2 hold-out roster (§7.1).
PHASE2_HOLDOUT_GAMES: tuple = (
    "twenty_forty_eight",
    "super_mario",
    "gymv_streets_of_rage_2",
    "gymv_space_harrier_ii",
    "gymv_airstriker",
    "gymv_strider",
)

# Evaluation-only: rollouts collected for metrics but NOT fed into GRPO training.
EVAL_ONLY_GAMES: List[str] = []

GAME_MAX_STEPS: Dict[str, int] = {
    "twenty_forty_eight": 200,
    "tetris": 200,
    "candy_crush": 50,
    "super_mario": 500,
    # Gym-V Temporal/* @ frame_skip=8.  200 agent steps ≈ 1600 emulator
    # frames ≈ 27 s of in-game time at 60 Hz, which comfortably covers
    # the ~100-step paper Table-3 anchor episodes for all 8 games while
    # still bounding the worst case (long Streets-of-Rage 2 stages).
    "gymv_thunder_force_iii": 200,
    "gymv_altered_beast": 200,
    "gymv_columns": 200,
    "gymv_dynamite_headdy": 200,
    "gymv_space_harrier_ii": 200,
    "gymv_streets_of_rage_2": 200,
    "gymv_airstriker": 200,
    "gymv_strider": 200,
}

EMULATOR_GAMES: set = set()


# ── Game-specific action priors (critical actions) ────────────────────
# Maps game → ordered list of action names the policy *must* use to make
# scoring progress.  These are surfaced in two places:
#
#   1. The action-selection prompt as a one-line hint, so the LLM sees
#      the prior in-context (no shaping reward — keeps GRPO advantages
#      from being further dominated by intrinsic).
#   2. ``_apply_anti_repetition`` substitutes the first critical action
#      that hasn't fired recently when the policy is stuck on a single
#      non-scoring action — this is a hard-coded escape valve only,
#      not a permanent reward signal.
#
# Why TF3 / Altered Beast etc. need this: post-mortem of the Phase-1
# collapse showed the action_taking policy was choosing UP/RIGHT 30-46%
# of the time but the B (fire/attack) button only ~3-7% of the time.
# In Sega-style horizontal shooters and brawlers the only way to score
# is sustained B presses; without an explicit prior the GRPO signal is
# too weak (raw env reward zero on 83% of TF3 episodes) to teach the
# model that B is the scoring action.  Adding a single-line schema hint
# is a much cheaper fix than waiting 50+ steps for GRPO to converge to
# the action vocab on its own.
#
# Action names match those exposed by the env's ``action_names`` list.
# Names that don't appear in a given episode's available actions are
# silently dropped at consumption time.  Keep the list short — only
# truly indispensable actions (one or two per game).
GAME_CRITICAL_ACTIONS: Dict[str, List[str]] = {
    # Horizontal-scrolling shoot-em-ups: B = primary fire.
    "gymv_thunder_force_iii":  ["B"],
    "gymv_space_harrier_ii":   ["B"],
    "gymv_airstriker":         ["B"],
    # Side-scrolling brawlers / action: B = attack.
    "gymv_altered_beast":      ["B"],
    "gymv_streets_of_rage_2":  ["B"],
    "gymv_strider":            ["B"],
    # gymv_dynamite_headdy: head-throw is mapped to B in the gymv
    # adapter; A is jump.
    "gymv_dynamite_headdy":    ["B"],
    # Falling-block puzzle: rotation matters but no single button is
    # indispensable. Intentionally omitted.
    # "gymv_columns": [],
}

# ── Multi-role rollout constants ─────────────────────────────────────
# When ``unified_role_rollouts`` is enabled, the same decision agent
# plays ALL roles and each rollout is tagged with role / side metadata.
# Per-game episode overrides ensure sufficient role coverage.

EPISODES_PER_GAME_MULTIROLE: Dict[str, int] = {}


# ── High-variance per-step episode overrides ─────────────────────────
# Empirical analysis of run ``Qwen3.5-9B_20260504_144712`` showed
# that the gymv shooter / brawler / scrolling-action subset is
# bimodal (success rate ~17% on TF3, ~12% on Altered Beast) which
# at the default n=8 episodes/step makes ``mean_reward`` jitter
# between 0 and 60+ purely from sampling noise — the apparent
# "collapse" at TF3 phase-1 steps 9-13 was almost entirely sampling
# variance with one process restart on top.  A 16-episode batch is
# the smallest n that drops the P(zero-mean | bimodal-success) noise
# floor below ~5% (bootstrap from the empirical TF3 distribution
# yields P=22.6% at n=8 and P=4.4% at n=16; see the run summary in
# the chat thread for the bootstrap calculation).
#
# We deliberately don't override every game — Tetris / Candy Crush
# are dense-reward and don't suffer the bimodal pathology; the
# paper Table-3 anchor games stay on the global default to avoid
# drifting away from the published numbers.  Adjust this dict (or
# pass ``--episodes-per-game-overrides '{...}'`` from the launcher)
# when adding new sparse-reward games.
HIGH_VARIANCE_GYMV_EPISODES: Dict[str, int] = {
    "gymv_thunder_force_iii":  16,
    "gymv_altered_beast":      16,
    "gymv_dynamite_headdy":    16,
    "gymv_space_harrier_ii":   16,
    "gymv_streets_of_rage_2":  16,
    "gymv_strider":            16,
    "gymv_airstriker":         16,
    # gymv_columns is dense-scoring (every match scores) so the
    # default 8 is enough — kept off the override list intentionally.
}


def resolve_bank_key(game: str, role: str = "", side: str = "") -> str:
    """Return the skill-bank routing key for an episode.

    In unified-role mode the key may encode a role dimension. For standard
    games the key is just the game name.

    Examples::

        resolve_bank_key("tetris") -> "tetris"
    """
    return game


def bank_keys_for_game(game: str) -> List[str]:
    """Return all possible bank keys for a game.

    Used by ``PerGameSkillBankManager`` to pre-create sub-bank pipelines.
    """
    return [game]


GAME_DURATION_ORDER = [
    "twenty_forty_eight",
    "tetris",
    "candy_crush",
    "super_mario",
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
    10: ["twenty_forty_eight", "tetris", "candy_crush", "super_mario"],
}

CURRICULUM_FOCUSED: Dict[int, List[str]] = {
    0: ["twenty_forty_eight", "tetris", "candy_crush", "super_mario"],
}

CURRICULUM_PRESETS: Dict[str, Optional[Dict[int, List[str]]]] = {
    "gradual": CURRICULUM_GRADUAL,
    "focused": CURRICULUM_FOCUSED,
    "none": None,
}


def _model_short_name(model_name: str) -> str:
    """Extract a filesystem-safe short name from a model identifier.

    ``"Qwen/Qwen3.5-9B"`` → ``"Qwen3.5-9B"``
    ``"meta-llama/Llama-3-8B"`` → ``"Llama-3-8B"``
    """
    short = model_name.rsplit("/", 1)[-1]
    return re.sub(r"[^\w\-.]", "_", short)


def _generate_run_dir(model_name: str) -> str:
    """Generate a unique run directory name from model name + timestamp.

    Example: ``runs/Qwen3.5-9B_20260427_131800``
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
    # When True, per-game episode counts follow ``episodes_per_game_overrides``.
    # When False (default), ``episodes_per_game_overrides`` ALSO applies
    # so individual high-variance games can be bumped without scaling
    # the global batch (bootstrapped from
    # :data:`HIGH_VARIANCE_GYMV_EPISODES` by default).
    unified_role_rollouts: bool = False
    episodes_per_game_overrides: Dict[str, int] = field(
        default_factory=lambda: {
            **EPISODES_PER_GAME_MULTIROLE,
            **HIGH_VARIANCE_GYMV_EPISODES,
        },
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

    # vLLM inference — base model that GRPO trains and that vLLM serves.
    # ``Qwen/Qwen3.5-9B`` is the multimodal hybrid Gated-DeltaNet + Gated-Attention
    # text decoder we LoRA-tune on.  For text-only rollouts the vision tower
    # is skipped via ``--language-model-only`` (see vllm_server.py).
    model_name: str = "Qwen/Qwen3.5-9B"
    temperature: float = 0.5
    max_tokens: int = 512
    vllm_base_url: str = "http://localhost:8000/v1"  # used only when manage_vllm=False
    vllm_base_port: int = 8000
    vllm_gpu_util: float = 0.95

    # Speculative decoding.  Two methods are supported:
    #   "mtp"          — use the model's built-in Multi-Token-Prediction head
    #                    (Qwen3.5-9B / Qwen3.5-MoE / Qwen3-Next ship one).
    #                    No draft model needed.  Recommended for Qwen3.5.
    #   "draft_model"  — use a small external model (e.g. Qwen3-0.6B) as the
    #                    drafter.  Required for plain Qwen3 (no MTP head).
    speculative_method: str = "mtp"
    speculative_model: Optional[str] = None
    num_speculative_tokens: int = 1

    # Inference-only secondary model (e.g. ``Qwen/Qwen3.5-35B-A3B``) — NOT
    # trained, served separately for evaluation, baselines, or as a teacher.
    # See ``inference/serve_qwen35_35b_a3b.sh`` for the standalone launcher;
    # this field is purely informational so config snapshots record which
    # inference model the run was paired with.
    inference_only_model: Optional[str] = "Qwen/Qwen3.5-35B-A3B"

    # When True, the orchestrator manages vLLM server lifecycle
    # (persistent instances on vllm_gpu_ids, hot-reload after GRPO).
    manage_vllm: bool = True

    # External opponent API (reserved for future multi-agent games).
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

    # ── Crafter + Promotion online hooks (Phase 1, off by default) ─────
    # Spec: implementation_notes/legacy/harness-usability-and-intra-gymv-transfer.md §3
    # When enabled, after every co-evolution step's Phase B finalize() the
    # trainer additionally runs:
    #   1. ``trainer.coevolution._crafter_hook.run_crafter_step`` —
    #      synthesizes FailureTraces from the just-finished EpisodeResults,
    #      calls SkillCrafterService.reflect_on_episode, dumps proposals
    #      in the offline-mirror JSONL schema.
    #   2. ``trainer.coevolution._promotion_hook.run_promotion_step`` —
    #      subprocess-invokes ``decide_promotion_gpt54.py --gate-mode
    #      offline-synthetic`` over those proposals, then writes the
    #      promoted (PROVISIONAL) skills back into each per-game
    #      ``skill_bank.jsonl`` via ``skill_bank.legacy_writeback``.
    # The hook is one-way (D8 Option A): the legacy 4-stage skill_agents
    # pipeline keeps writing to per-game banks unchanged, the new path
    # only *appends* promoted skills.
    crafter_promotion_enabled: bool = False
    # Per-batch Crafter cycle cadence. ``0`` disables cycle() entirely
    # (per-episode reactive pass still runs).  Recommend 5-10 in steady
    # state — runs Composer / Generalizer over accumulated failures.
    crafter_cycle_every_k_steps: int = 0
    # Total reward at/below this threshold counts as OUTCOME_FAILURE.
    # ``0.0`` matches the offline mirror's default.
    crafter_outcome_failure_threshold: float = 0.0
    # Hard wall-clock cap on the decide_promotion subprocess invocation.
    # Phase-0 Airstriker baseline is ~0.3s; 300s leaves room for the
    # 13-game sweep + future Stage-1 replay overhead.
    crafter_promotion_timeout_s: float = 300.0
    # Gate mode forwarded to ``decide_promotion_gpt54.py --gate-mode``.
    # ``offline-synthetic`` (default) keeps Phase-1's documented
    # behaviour: rule-only Stage 0 + LIMITED_PASS placeholders for
    # Stages 1-4, no LLM calls. ``offline-with-llm-judge`` extends it
    # with one ``BACKBONE_JUDGE_MODEL`` (35B-A3B) call per proposal —
    # the 35B verdict is appended as an extra StageVerdict and an LLM
    # ``fail`` flips the aggregate to FAIL ⇒ REJECT, so visibly bad
    # proposals are filtered out before they enter the bank. The
    # 35B endpoint is selected via ``VLLM_BASE_URL_MAP`` (handled by
    # ``API_func``); no extra plumbing required from the trainer side.
    crafter_promotion_gate_mode: str = "offline-synthetic"
    # Stage 2 cross-domain knobs for the LLM judge that runs inside
    # ``offline-with-llm-judge`` mode (no effect for any other gate
    # mode).  When ``crafter_promotion_judge_enable_thinking=True``
    # the orchestrator forwards ``--enable-thinking`` to
    # ``decide_promotion_gpt54.py``, which threads through to
    # ``_llm_skill_judge.judge_proposal``'s ``ask_model`` call.  We
    # also expose ``crafter_promotion_judge_max_tokens`` separately
    # so callers can pair the flag with the budget the ``<think>``
    # block actually needs.
    #
    # Live probes against Qwen3.5-35B-A3B (judge prompt ~620 input
    # tokens) observed the model spend ~5K tokens on the ``<think>``
    # block before emitting the ~120-char JSON verdict.
    # ``max_tokens=4096`` truncates inside the think block;
    # ``max_tokens=8192`` completes in ~25-40 s; ``max_tokens=16384``
    # is comfortable headroom.  Stage-1 in-domain training keeps
    # both at their fast-path defaults (thinking off, 256 tokens /
    # verdict ⇒ no <think> block, ~0.2 s / call).
    crafter_promotion_judge_enable_thinking: bool = False
    crafter_promotion_judge_max_tokens: int = 256
    # Lane-(a) feature flag. ``False`` (default) parks the Repairer /
    # PatchProposal mint path: under
    # ``implementation_notes/legacy/skill-lane-decision.md`` skills are
    # retrieval payloads, not runnable programs, so live protocol-edit
    # proposals would be edits to a contract no live runtime executes.
    # The Crafter dispatcher's existing ``_STATUS_NO_OP`` →
    # Hypothesizer fall-through carries the failure signal through.
    # Set ``True`` only when running explicit lane-(b) experiments.
    crafter_enable_protocol_patching: bool = False

    # ── Cross-domain transfer gate (Layer A) ────────────────────────
    # When ``True`` (and ``crafter_promotion_enabled=True``), the
    # trainer inserts ``trainer.coevolution._transfer_hook.run_transfer_gate_step``
    # after the promotion writeback. The gate re-evaluates each
    # just-promoted skill's cross-domain admit rate against the
    # configured ``crafter_transfer_targets`` and rolls back
    # promotions that fall below ``crafter_transfer_admit_band[0]``
    # on every target.
    #
    # The matrix subprocess is the same
    # ``labeling_supplement/_phase4_transfer_matrix.py`` driver the
    # Stage-6 measurement uses. Per-step gating only makes sense when
    # ``crafter_promotion_enabled`` already runs at low cadence (e.g.
    # every K steps); see
    # ``implementation_notes/coevolution-cross-domain-integration.md``
    # §4 for the full contract + acceptance criteria.
    crafter_transfer_gate_enabled: bool = False
    # Target corpora the matrix driver evaluates against. Subset of
    # the canonical Stage-6 cluster set (video / visual_reasoning /
    # osworld / browser). Empty list disables the gate at runtime
    # without flipping the master flag.
    crafter_transfer_targets: Tuple[str, ...] = ("video", "visual_reasoning")
    # ``(lower, upper)`` admit-rate band per §11.5.4 of the cross-
    # domain measurement plan. Skills failing ``admit_rate < lower``
    # on every target are DEMOTED. ``upper`` is informational (Layer
    # D dashboard surfaces it for the Generalizer's signal).
    crafter_transfer_admit_band: Tuple[float, float] = (0.15, 0.60)
    # Minimum number of targets a skill must clear ``band[0]`` on to
    # KEEP. ``1`` is the most permissive: a single transferable
    # target is enough to keep the promotion.
    crafter_transfer_min_targets_in_band: int = 1
    # Forwarded to ``_phase4_transfer_matrix.py --max-skills``. The
    # synthetic bank-run is already filtered to just-promoted skills
    # so this caps the per-cell sweep within that subset.
    crafter_transfer_max_skills_per_cell: int = 5
    # Hard wall-clock cap on the matrix subprocess. The driver loads
    # 1-2 demos per cell; conda-env helpers (browser) cold-boot
    # adds ~30s; 1800s is a generous ceiling for K=4 targets × N=5
    # skills.
    crafter_transfer_timeout_s: float = 1800.0

    # ── Cross-domain dashboard (Layer D) ────────────────────────────
    # Periodic offline pass that snapshots the trainer's banks and
    # runs the full N×N Stage-6 transfer matrix to surface the live
    # cross-domain transfer health (G1-G5 acceptance gates, per-
    # cluster admit rates) to wandb / TensorBoard. Decoupled from
    # the per-step transfer gate (Layer A) so dashboards can run at
    # low frequency (every 100 steps) while the gate runs at
    # higher frequency without doubling the subprocess load.
    #
    # Spec:  implementation_notes/coevolution-cross-domain-integration.md §5
    crafter_dashboard_enabled: bool = False
    # Cadence: dashboard fires when ``step % crafter_dashboard_every_k_steps == 0``.
    # ``0`` disables the dashboard via cadence (master flag stays
    # untouched). Recommend 100-500 in steady state — a full N×N
    # sweep takes ~1-30 minutes depending on bank size + N targets.
    crafter_dashboard_every_k_steps: int = 0
    # Target corpora the dashboard's matrix sweep evaluates against.
    # Typically the full canonical Stage-6 set so the wandb dashboard
    # shows complete G3-G5 verdicts.
    crafter_dashboard_targets: Tuple[str, ...] = (
        "video", "visual_reasoning", "osworld", "browser",
    )
    # Forwarded to ``_phase4_transfer_matrix.py --max-skills``.
    crafter_dashboard_max_skills_per_cell: int = 5
    # Hard wall-clock cap on the matrix subprocess. Default 1h leaves
    # room for the full N×N sweep with conda-env helpers (browser)
    # cold-booting.
    crafter_dashboard_timeout_s: float = 3600.0

    # ── Harness wire-up (Day-10) ────────────────────────────────────
    # When enabled, the trainer wires the harness's two LLM-free
    # surfaces into Phase A rollouts:
    #   1. ``SkillHarness.select_eligible_skills`` filters the cold-
    #      start RAG candidates *before* the skill_selection LLM picks.
    #   2. ``SkillHarness.validate_invocation`` second-pass-vetoes the
    #      LLM's chosen skill, falling back to the next eligible one
    #      when vetoed.
    # The aggregated rejection patterns ride into Phase B′ via
    # ``RejectedSkillSink → SkillLifecycleManager.record_false_binding_pattern``
    # so the Crafter's Repairer sees richer evidence on each
    # ``SkillRecord.false_binding_patterns`` (PLAN-SKILL-BANK §4.3b).
    # See ``trainer/coevolution/_harness_hook.py`` for the full
    # contract; ``harness/README.md`` §22 for the spec gap this closes.
    harness_enabled: bool = False
    # Whether SHADOW skills are admitted by the eligibility filter.
    # Trainer default ``True`` mirrors ``HarnessConfig.allow_shadow``.
    # Disable with ``--no-harness-allow-shadow`` for runs that must
    # bind only fully-validated (ACTIVE / PROVISIONAL) skills.
    harness_allow_shadow: bool = True

    # ── Block B1 — harness ablation mode ──────────────────────────────
    # Three-way switch for the §5.5 "w/o harness" ablation:
    #   * ``"full"`` — eligibility filter + validate_invocation +
    #     adaptation scoring fully active (default; same as
    #     ``harness_enabled=True`` historically).
    #   * ``"plain-text-skills"`` — skill candidates are still surfaced
    #     to the actor (so few-shot prompt content matches the SkillBridge
    #     run), but the eligibility filter and validate_invocation are
    #     bypassed; the actor sees raw skill-bank content with no
    #     grounding / precondition / admissibility check.  This is the
    #     reviewer-requested "skills as plain-text few-shot" baseline.
    #   * ``"off"`` — no skill candidates at all (cold-start mode).
    # When ``harness_enabled=False`` this knob is ignored.
    harness_mode: str = "full"

    # ── Block B2 — w/o crafter ablation ───────────────────────────────
    # When ``crafter_enabled=False``, the orchestrator skips
    # ``run_crafter_step`` entirely (no patches, no LLM crafter, no
    # hypotheses, no retire proposals).  Promotion + lifecycle still
    # run on top of any pre-existing draft skills (so the bank can
    # still grow from the legacy Stage-4 curator), but no new
    # mutation evidence is produced.  This isolates the §4.3 crafter
    # contribution from the §4.4 lifecycle.  Implies
    # ``crafter_promotion_enabled=True`` (the cycler still runs in
    # promotion mode).
    crafter_enabled: bool = True

    # ── Block B3 — promotion gate bypass ──────────────────────────────
    # ``"gated"`` (default) routes proposals through
    # ``GateService`` (offline-synthetic / offline-with-llm-judge / live).
    # ``"permissive"`` auto-approves every proposal — DRAFT → ACTIVE
    # straight through with no judge call.  Used to measure the
    # §4.4 lifecycle gate contribution in isolation.
    promotion_bypass_mode: str = "gated"

    # ── Block B4 — intention trigger ablation ─────────────────────────
    # Controls *when* a fresh ``intention`` LLM call happens inside the
    # episode loop:
    #   * ``"every-step"`` (default, matches historical SkillBridge
    #     production behaviour pre-block-B): re-generate every inner
    #     step.  Don't change the default without rerunning the full
    #     §5.5 baseline.
    #   * ``"sharp-shift"`` — re-generate only when the textual delta
    #     or urgency signal indicates a meaningful state change.
    #   * ``"disabled"`` — never re-generate; actor reuses the first
    #     intention for the whole episode.
    intention_trigger: str = "every-step"

    # ── Block B5 — actor bank cap ─────────────────────────────────────
    # Top-K cap applied at the actor's skill-query surface.  When
    # ``actor_bank_cap_k > 0``, the actor sees only the top-K active
    # skills (ranked by retrieval score).  ``0`` (default) disables
    # the cap — historical behaviour, actor sees the full active bank.
    # Used for the bank-size sweep K ∈ {10, 50, 200}.
    actor_bank_cap_k: int = 0

    # ── Path 2 — supplemental LLM Crafter (35B) ─────────────────────
    # When enabled, ``_crafter_hook.run_crafter_step`` augments its
    # rule-based proposals with up to ``llm_crafter_k_max`` additional
    # 35B-A3B-driven proposals per game per step (one call per
    # FailureTrace, parallelised via asyncio.gather, fail-soft on
    # any error).  Routes through ``API_func.ask_model`` →
    # ``BACKBONE_JUDGE_MODEL`` (or ``llm_crafter_model`` if non-empty)
    # via ``VLLM_BASE_URL_MAP`` — same plumbing as the LLM promotion
    # judge.  See ``trainer/coevolution/_llm_crafter.py`` and
    # ``crafter-harness-orchestrator-roles.md`` §2.1.
    llm_crafter_enabled: bool = False
    # Empty → defer to ``BACKBONE_JUDGE_MODEL`` (35B-A3B teacher).
    llm_crafter_model: str = ""
    # Hard cap on parallel 35B calls per game per step. Lowered from 5
    # → 2 in the post-v11 fix (see ``crafter_hypothesize_*`` block
    # above): with the rewritten last-resort prompt and the recurrence
    # / relatedness gates, a small per-step volume cap is safe and
    # keeps token budget predictable.  Override via
    # ``--llm-crafter-k-max`` (env: ``LLM_CRAFTER_K_MAX``).
    llm_crafter_k_max: int = 2
    # Token budget per LLM Crafter response.
    llm_crafter_max_tokens: int = 1024
    # Sampling temperature for LLM Crafter calls.
    llm_crafter_temperature: float = 0.3
    # Hard wall-time per individual 35B call.  On timeout we drop the
    # one trace and continue; a slow 35B can never block a step.
    llm_crafter_timeout_s: float = 60.0
    # Stage 2 (cross-domain adaptation) opt-in.  When ``True``, the
    # 35B Crafter calls in ``_crafter_hook`` forward
    # ``enable_thinking=True`` into ``API_func.ask_vllm`` so Qwen3-A3B
    # emits its ``<think>`` chain-of-thought before the final JSON
    # proposal.  In Stage 1 (in-domain curriculum, all 6 phases of
    # ``run_phase1_curriculum.sh``) this stays ``False`` because the
    # rule-based proposals already cover the easy patches and the
    # extra wall-time would dominate per-step latency.
    #
    # EXPERIMENTAL: live probes against Qwen3.5-35B-A3B observed the
    # Crafter prompt induce >16K-token ``<think>`` blocks that never
    # emitted a final JSON answer (the prompt's open-ended
    # patch / hypothesize / retire dispatch invites runaway
    # reasoning).  Stage-2 callers that flip this to ``True`` SHOULD
    # therefore (a) bump ``llm_crafter_max_tokens`` to ≥ 16384,
    # (b) bump ``llm_crafter_timeout_s`` to ≥ 180, AND (c) re-tune
    # ``_llm_crafter._build_prompt`` to constrain reasoning length
    # ("think briefly", explicit JSON-first instruction, etc.)
    # before relying on the path.
    llm_crafter_enable_thinking: bool = False

    # ── Hypothesizer fallthrough gate (post-v11 audit) ──────────────
    # The crafter dispatch chain is `patch → retire → hypothesize`,
    # with hypothesize as last-resort.  In v11 the trigger conditions
    # were too loose: a single orphan failure (skill_id missing) fell
    # straight through to the Hypothesizer, which minted an empty
    # placeholder skill on every episode.  Result: 73-85% of the bank
    # was boilerplate ``hypothesis__prop-...`` records that the actor
    # never selected, contract GRPO collapsed (effect-literal learning
    # rate fell from 70-85% → 6%), and TF3 step-0 reward dropped from
    # 688 (5/3 baseline) to 62.
    #
    # Two gates restore the architectural intent that hypothesize fires
    # only on genuinely hard cases:
    #
    #   * ``crafter_hypothesize_min_recurrences`` — same failure
    #     pattern must recur ≥ N times in the FailureMemory window.
    #     Default 3 mirrors the per-batch ``hot_pattern_threshold``,
    #     so per-episode and per-batch dispatch share a recurrence
    #     bar for the *new-skill* exit.  Set to 1 to reproduce v11
    #     behaviour (test-only knob).
    #   * ``crafter_hypothesize_related_skill_jaccard`` — minimum
    #     token-Jaccard overlap (skill_id + name + strategic_description
    #     vs failure pattern signature + diagnosis + abort_reason)
    #     above which the gate decides "a related skill already
    #     exists, prefer patch".  Default 0.30 — the same threshold
    #     ``skill_agents.query._compute_relevance`` uses for
    #     retrieval relevance, so the gate's notion of "related"
    #     matches the actor's downstream selection signal.  Set
    #     to 0.0 to disable the relatedness gate (recurrence gate
    #     stays).
    crafter_hypothesize_min_recurrences: int = 3
    crafter_hypothesize_related_skill_jaccard: float = 0.30

    # ── Reviewer-facing instrumentation (block A) ─────────────────────
    # Enable per-event JSONL streams used by the §4.3 / §5.3 / §5.5
    # analysis scripts:
    #   * ``run_dir/harness_log/rejections.jsonl`` — every veto code +
    #     domain + skill_id (drives failure-mode pie chart).
    #   * ``run_dir/harness_log/validate.jsonl`` — every per-event
    #     ``validate_invocation`` diagnostic (drives case studies).
    #   * ``run_dir/lifecycle_log/transitions.jsonl`` — every
    #     ``SkillStatus`` transition (drives lifetime distribution +
    #     promotion/deprecation curves).
    #   * ``run_dir/intention_log/switches.jsonl`` — every per-step
    #     intention update (drives intention-trigger ablation).
    #   * ``run_dir/runtime_log/component_timings.jsonl`` — per-component
    #     vLLM call counts + latency (drives runtime overhead Q8).
    # I/O cost is ~1MB/step for a 6-game phase; disable for
    # latency-sensitive ablation runs.
    reviewer_instrumentation_enabled: bool = True

    # ── Path 4 — LLM Harness validator (35B) ────────────────────────
    # Post-LLM second-pass validation by the 35B-A3B teacher,
    # complementing the deterministic
    # :meth:`SkillHarness.validate_invocation`.  Hybrid policy:
    #   * Bootstrap window: when ``trainer_step <
    #     llm_harness_bootstrap_steps`` the LLM validator fires on
    #     EVERY admitted skill.
    #   * Steady state: only fires when the deterministic verdict
    #     was uncertain (e.g. SHADOW skill, no can_handle evidence,
    #     translation-rewritten contract).
    # Verdicts can ONE-WAY downgrade admit→veto; they never
    # override a deterministic veto upward.  Episode-level cache
    # keyed by ``(episode_id, skill_id)`` so repeated picks of the
    # same skill in one episode pay the LLM cost only once.  See
    # ``trainer/coevolution/_llm_harness_validator.py``.
    llm_harness_validator_enabled: bool = False
    # Empty → defer to ``BACKBONE_JUDGE_MODEL`` (35B-A3B teacher).
    llm_harness_model: str = ""
    # Bootstrap window: outer-loop training steps below this value
    # always go through the LLM validator regardless of deterministic
    # certainty.  Default 20 ≈ ~1 day of P1 training.
    llm_harness_bootstrap_steps: int = 20
    # Token budget per LLM Harness validator response.
    llm_harness_max_tokens: int = 256
    # Sampling temperature for LLM Harness validator calls.
    llm_harness_temperature: float = 0.2
    # Hard wall-time per individual 35B call.  On timeout the
    # deterministic verdict stands (admit).
    llm_harness_timeout_s: float = 30.0

    # ── Phase-start GameProfile (Path 1) ────────────────────────────
    # When ``True``, the orchestrator runs ``ensure_game_schemas`` once
    # per curriculum phase boundary (and at startup) and injects a
    # compact GameProfile (goal / win_signal / hazards / key_actions /
    # failure_modes) into the actor's SYSTEM_PROMPT and
    # SKILL_SELECTION_SYSTEM_PROMPT for the duration of the phase.
    # Adds 1 × ``BACKBONE_JUDGE_MODEL`` (35B-A3B) call per game per
    # phase boundary, no per-step LLM cost. The same call also produces
    # a ``<state>...</state>`` exemplar cached under
    # ``run_dir/phase_artifacts/<game>.schema.json`` for Path 2 (LLM
    # Crafter) and Path 4 (LLM Harness validator) to reuse as a
    # few-shot anchor without firing their own 35B calls.
    # See ``trainer/coevolution/_game_schema.py``.
    game_schema_enabled: bool = False
    # Override model for GameProfile generation; empty string defers to
    # ``BACKBONE_JUDGE_MODEL`` resolved by ``API_func.ask_model`` via
    # ``VLLM_BASE_URL_MAP`` (same routing as the LLM promotion judge).
    game_schema_model: str = ""
    # Token budget for the 35B response (compact GAME_PROFILE block +
    # full STATE_EXAMPLE block fit comfortably in ~1k tokens).
    game_schema_max_tokens: int = 1024
    # Hard timeout per 35B call. On timeout we fall back to the
    # deterministic minimum profile and continue without blocking the
    # phase boundary.
    game_schema_timeout_s: float = 60.0

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

    # T2.4 — single reward sink. When set (auto-resolved by
    # ``resolve_paths`` to ``{rewards_dir}/reward_log.jsonl`` if left
    # empty), the orchestrator constructs a ``harness.RewardLogger``
    # against this path and threads it through every rollout. Each
    # ``GRPORecord`` written by the trainer is mirrored into the same
    # JSONL (``kind="grpo_step"``) so eval and training read from one
    # source. Set to ``""`` to disable.
    reward_log_path: str = ""

    # T2.7 — curator overfit mitigation. The CURATOR LoRA receives a
    # scaled GRPO reward equal to ``base * min(1, step / warmup) *
    # curator_weight``. With ``curator_warmup_steps=0`` (default) the
    # ramp is disabled and the scalar reward passes through unchanged
    # (``curator_weight`` still applies as a constant). Set
    # ``curator_warmup_steps`` to a small positive integer (e.g. 50)
    # to dampen early-training noise; set ``curator_weight`` < 1.0
    # for a permanent down-weight (the curator becomes a tie-breaker
    # rather than a hard gate). Wired via
    # ``skill_agents.bank_maintenance.llm_curator.set_curator_warmup``
    # at the start of every outer-loop step.
    curator_weight: float = 1.0
    curator_warmup_steps: int = 0

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

    # Storage layout for the skill bank across curriculum games.
    #
    # ``"per_game"`` (default, legacy):
    #     One ``<bank_dir>/<game>/skill_bank.jsonl`` per game. Skills
    #     never cross game boundaries on disk; cross-game effects only
    #     accumulate at the LoRA-weight level.
    #
    # ``"shared"`` (cross-game lifelong-learning experiments):
    #     One ``<bank_dir>/skill_bank.jsonl`` shared across all games.
    #     Eligibility is enforced at runtime by the harness's
    #     ``EligibilityFilter`` task-axis veto (F2′,
    #     ``harness/README §22``) — every skill carries
    #     ``feasible_tasks=[<game>]`` so SoR2-mined skills only fire on
    #     SoR2 states unless the cross-game translator
    #     (``skill_agents.skill_bank.translate_for_target``) emits an
    #     explicit derived record. Pair with
    #     ``--seed-bank-dir`` and the per-phase translation step in
    #     ``scripts/run_phase1_curriculum.sh`` for the full lifelong
    #     pipeline.
    #
    # Default ``"per_game"`` preserves every prior run's behaviour
    # bit-for-bit; the shared path is opt-in via
    # ``run_coevolution.py --bank-mode shared``.
    bank_mode: str = "per_game"

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

    # LoRA adapter defaults (matches skill_agents.lora.config)
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

    _resolved: bool = field(default=False, repr=False)

    def resolve_paths(self) -> "CoEvolutionConfig":
        """Rebase all directory paths under ``run_dir``.

        If ``run_dir`` is ``None``, generates one from the model name
        and current timestamp (e.g. ``runs/Qwen3.5-9B_20260427_131800``).

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
        # T2.4 — auto-place the unified reward log under ``rewards_dir``
        # when the field was left at its default empty string. Explicit
        # ``""`` after ``resolve_paths()`` means the user disabled the
        # sink (already-resolved paths are not re-rebased).
        if self.reward_log_path == "":
            self.reward_log_path = str(Path(self.rewards_dir) / "reward_log.jsonl")
        elif not Path(self.reward_log_path).is_absolute():
            self.reward_log_path = str(root / self.reward_log_path)

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

        Per-game overrides apply in BOTH modes:
          * Unified-role mode: covers role-coverage fan-out (5 for Avalon,
            7 for Diplomacy by default).
          * Standard per-game mode: covers high-variance sparse-reward
            games (gymv shooters / brawlers default to 16 to keep the
            sampling-noise floor below ~5%; see
            :data:`HIGH_VARIANCE_GYMV_EPISODES`).
        Games not in the override dict use the global ``episodes_per_game``.
        """
        return self.episodes_per_game_overrides.get(game, self.episodes_per_game)

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
    try:
        # transformers 5.x exposes the multimodal "image-text-to-text" auto class
        # used by Qwen3.5 / Qwen3-VL.  Fall back gracefully for text-only setups.
        from transformers import AutoModelForImageTextToText  # type: ignore
    except ImportError:  # pragma: no cover — older transformers
        AutoModelForImageTextToText = None  # type: ignore

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
    model_cfg = AutoConfig.from_pretrained(
        config.model_name, trust_remote_code=True,
    )
    # Multimodal Qwen3.5 / Qwen3-VL configs nest the language-model spec under
    # ``text_config``.  Use it for arch-detection so we identify the LM type
    # (e.g. ``qwen3_5_text``) rather than the umbrella multimodal type.
    text_cfg = getattr(model_cfg, "text_config", model_cfg)
    text_arch = (getattr(text_cfg, "model_type", "") or "").lower()
    is_multimodal = hasattr(model_cfg, "text_config") or hasattr(model_cfg, "vision_config")

    # ── T2.11 closure: single-source-of-truth target_modules resolver ──
    # SFT and GRPO must write/read the same LoRA shape, otherwise legs
    # missing in one recipe silently drop deltas at the boundary.  We
    # therefore delegate to ``trainer.SFT.lora_targets`` for both
    # pipelines — see ``implementation_notes/pre-training-readiness-audit.md``
    # §0.3.  Qwen3.5 hybrid stack reaches ALL GatedDeltaNet legs incl.
    # ``in_proj_z/b/a`` (``in_proj_z`` is hidden×value_dim, NOT tiny — the
    # earlier "skip the gating legs" rationale undercounted it).
    from trainer.SFT.lora_targets import resolve_target_modules as _resolve_targets

    target_modules = _resolve_targets(
        text_arch=text_arch,
        explicit=config.lora_target_modules,
    )

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
        "[init=%s, r=%d, alpha=%d, arch=%s, multimodal=%s]: %s",
        config.model_name, len(need_init), init_desc,
        config.lora_r, config.lora_alpha,
        text_arch or "unknown", is_multimodal, need_init,
    )
    # For multimodal Qwen3.5 / Qwen3-VL the umbrella ``Qwen3_5Config`` lacks
    # ``vocab_size`` (lives under ``text_config``), so ``AutoModelForCausalLM``
    # crashes when fed the outer config.  Use ``AutoModelForImageTextToText``
    # in that case — the LoRA target_modules above still only match text
    # decoder linears, leaving the (frozen) vision tower untouched.
    if is_multimodal and AutoModelForImageTextToText is not None:
        loader_cls = AutoModelForImageTextToText
    else:
        loader_cls = AutoModelForCausalLM
    base_model = loader_cls.from_pretrained(
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
