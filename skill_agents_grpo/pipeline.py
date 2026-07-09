"""
SkillBank Agent — agentic pipeline that builds, maintains, and serves a Skill Bank.

Orchestrates boundary_proposal (Stage 1), infer_segmentation (Stage 2),
stage3_mvp (Stage 3 contract learn/verify/refine), bank_maintenance (Stage 4
split/merge/refine), and skill_evaluation.  Exposes a query API consumed by
decision_agents.

Typical usage::

    from skill_agents_grpo.pipeline import SkillBankAgent, PipelineConfig

    agent = SkillBankAgent(bank_path="skills/bank.jsonl")
    agent.ingest_episodes(episodes, env_name="llm+overcooked")
    agent.run_until_stable(max_iterations=3)

    # Decision-agent queries the bank
    result = agent.query_skill("navigate to pot and place onion")
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from skill_agents_grpo.skill_bank.bank import SkillBankMVP
from skill_agents_grpo.skill_bank.new_pool import (
    NewPoolManager,
    NewPoolConfig,
    ProtoSkillManager,
    ProtoSkillConfig,
)
from skill_agents_grpo.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Top-level configuration for the SkillBank Agent pipeline."""

    # Paths
    bank_path: Optional[str] = None
    preference_store_path: Optional[str] = None
    report_dir: Optional[str] = None

    # Stage 1: boundary proposal
    env_name: str = "llm"
    game_name: str = "generic"  # actual game identifier for phase detection
    merge_radius: int = 3
    extractor_model: Optional[str] = None

    # Stage 2: segmentation
    segmentation_method: str = "dp"
    preference_iterations: int = 3
    margin_threshold: float = 1.0
    max_queries_per_iter: int = 5
    new_skill_penalty: float = 2.0

    # Boundary scoring (tag-change penalty model)
    consistency_penalty: float = 0.3
    min_segment_length: int = 3
    min_skill_length: int = 2
    boundary_score_threshold: float = 0.3

    # Contract feedback: Stage 3 → Stage 2 closed loop
    contract_feedback_mode: str = "off"  # "off" | "weak" | "strong"
    contract_feedback_strength: float = 0.3

    # Stage 3: contract learning
    eff_freq: float = 0.8
    min_instances_per_skill: int = 5
    start_end_window: int = 5

    # NEW pool management
    new_pool_min_cluster_size: int = 5
    new_pool_min_consistency: float = 0.5
    new_pool_min_distinctiveness: float = 0.25

    # Stage 4: bank maintenance
    split_pass_rate_threshold: float = 0.7
    child_pass_rate_threshold: float = 0.8
    merge_jaccard_threshold: float = 0.85
    merge_embedding_threshold: float = 0.90
    min_child_size: int = 3
    min_new_cluster_size: int = 5

    # Convergence
    max_iterations: int = 5
    convergence_margin_std: float = 0.5
    convergence_new_rate: float = 0.05

    # LLM
    llm_model: Optional[str] = None
    max_concurrent_llm_calls: Optional[int] = None  # cap for local GPU (e.g. 1)
    # Max worker threads for parallel LLM calls within a single episode's
    # segmentation. Set to 1 when the caller already parallelises across
    # episodes (e.g. co-evolution loop) to avoid thread explosion.
    llm_teacher_max_workers: Optional[int] = None
    llm_teacher_max_tokens: Optional[int] = None  # override LLMTeacherConfig default (1000)


# ─────────────────────────────────────────────────────────────────────
# Pipeline state (serialisable snapshot)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IterationSnapshot:
    """Diagnostics captured after one pipeline iteration."""

    iteration: int = 0
    n_skills: int = 0
    n_segments: int = 0
    n_new_segments: int = 0
    new_rate: float = 0.0
    mean_margin: float = 0.0
    mean_pass_rate: float = 0.0
    n_splits: int = 0
    n_merges: int = 0
    n_refines: int = 0
    n_materialized: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────
# SkillBankAgent
# ─────────────────────────────────────────────────────────────────────

class SkillBankAgent:
    """Agentic pipeline that builds, maintains, and serves a Skill Bank.

    Lifecycle::

        agent = SkillBankAgent(...)
        agent.load()                         # load existing bank if any
        agent.ingest_episodes(episodes, ...)  # Stage 1+2+3 per episode
        agent.run_until_stable()              # iterate Stage 2→3→4 to convergence
        result = agent.query_skill("...")     # decision-agent queries

    The agent accumulates ``SegmentRecord`` objects across ingestion calls and
    uses them for bank maintenance and evaluation.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        bank_path: Optional[str] = None,
        bank: Optional[SkillBankMVP] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        if bank_path:
            self.config.bank_path = bank_path

        self.bank = bank or SkillBankMVP(path=self.config.bank_path)
        self._all_segments: List[SegmentRecord] = []
        self._new_pool: List[SegmentRecord] = []  # legacy list (kept for compat)
        self._new_pool_mgr = NewPoolManager(config=NewPoolConfig(
            min_cluster_size=self.config.new_pool_min_cluster_size,
            min_consistency=self.config.new_pool_min_consistency,
            min_distinctiveness=self.config.new_pool_min_distinctiveness,
            min_pass_rate=self.config.split_pass_rate_threshold,
        ))
        self._proto_mgr = ProtoSkillManager(config=ProtoSkillConfig())
        self._observations_by_traj: Dict[str, list] = {}
        self._predicates_by_traj: Dict[str, List[Dict[str, float]]] = {}
        self._traj_lengths: Dict[str, int] = {}
        self._preference_store: Any = None  # lazily created
        self._iteration: int = 0
        self._history: List[IterationSnapshot] = []
        self._query_engine: Any = None  # lazily created

    # ── Persistence ──────────────────────────────────────────────────

    def load(self) -> None:
        """Load skill bank (and preference store if path set)."""
        if self.config.bank_path:
            self.bank.load(self.config.bank_path)
            logger.info("Loaded bank with %d skills from %s", len(self.bank), self.config.bank_path)

        if self.config.preference_store_path:
            store = self._get_or_create_preference_store()
            store.load(self.config.preference_store_path)

    def save(self) -> None:
        """Persist bank, preference store, and iteration history."""
        if self.config.bank_path:
            self.bank.save(self.config.bank_path)
            logger.info("Saved bank (%d skills) to %s", len(self.bank), self.config.bank_path)

        if self.config.preference_store_path and self._preference_store is not None:
            self._preference_store.save(self.config.preference_store_path)

        if self.config.report_dir and self._history:
            report_dir = Path(self.config.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            history_path = report_dir / "iteration_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in self._history], f, indent=2, default=str)

    # ── Lazy helpers ─────────────────────────────────────────────────

    def _get_or_create_preference_store(self):
        if self._preference_store is None:
            from skill_agents_grpo.infer_segmentation.preference import PreferenceStore
            self._preference_store = PreferenceStore()
        return self._preference_store

    def _get_query_engine(self):
        if self._query_engine is None:
            from skill_agents_grpo.query import SkillQueryEngine
            self._query_engine = SkillQueryEngine(self.bank)
        return self._query_engine

    def _invalidate_query_engine(self):
        self._query_engine = None

    # Per-game default skill seeds: phase × relevant subgoal tags.
    # Mirrors the naming scheme in labeling/extract_skillbank_gpt54.py
    # so skill names are game-stage-aware from the first episode.
    _GAME_DEFAULT_SEEDS: Dict[str, List[str]] = {
        "tetris": [
            "opening:SETUP", "opening:CLEAR", "opening:POSITION",
            "midgame:SETUP", "midgame:CLEAR", "midgame:POSITION", "midgame:OPTIMIZE",
            "endgame:CLEAR", "endgame:SURVIVE", "endgame:OPTIMIZE",
        ],
        "twenty_forty_eight": [
            "opening:MERGE", "opening:SETUP", "opening:POSITION",
            "midgame:MERGE", "midgame:POSITION", "midgame:OPTIMIZE",
            "endgame:MERGE", "endgame:SURVIVE", "endgame:OPTIMIZE",
        ],
        "2048": [
            "opening:MERGE", "opening:SETUP", "opening:POSITION",
            "midgame:MERGE", "midgame:POSITION", "midgame:OPTIMIZE",
            "endgame:MERGE", "endgame:SURVIVE", "endgame:OPTIMIZE",
        ],
        "candy_crush": [
            "early:CLEAR", "early:SETUP", "early:POSITION",
            "mid:CLEAR", "mid:OPTIMIZE", "mid:EXECUTE",
            "late:CLEAR", "late:OPTIMIZE", "late:SURVIVE",
        ],
        "super_mario": [
            "early_level:NAVIGATE", "early_level:COLLECT", "early_level:ATTACK",
            "mid_level:NAVIGATE", "mid_level:SURVIVE", "mid_level:ATTACK",
            "late_level:NAVIGATE", "late_level:SURVIVE", "late_level:EXECUTE",
        ],
        "sokoban": [
            "explore:NAVIGATE", "explore:EXPLORE",
            "setup:POSITION", "setup:SETUP",
            "solving:NAVIGATE", "solving:EXECUTE",
            "finishing:EXECUTE", "finishing:OPTIMIZE",
        ],
        "avalon": [
            "early_quests:SETUP", "early_quests:DEFEND",
            "mid_quests:ATTACK", "mid_quests:DEFEND",
            "final_quest:EXECUTE", "final_quest:SURVIVE",
            "team_building:SETUP", "discussion:EXPLORE",
        ],
        "diplomacy": [
            "opening:SETUP", "opening:NAVIGATE", "opening:ATTACK",
            "orders:ATTACK", "orders:DEFEND", "orders:NAVIGATE",
            "retreat:DEFEND", "retreat:SURVIVE",
            "adjustment:BUILD", "adjustment:OPTIMIZE",
            "late_orders:ATTACK", "late_orders:DEFEND",
        ],
        "pokemon_red": [
            "battle:ATTACK", "battle:DEFEND", "battle:SURVIVE",
            "overworld:NAVIGATE", "overworld:EXPLORE", "overworld:COLLECT",
            "menu:SETUP", "menu:OPTIMIZE",
            "dialog:EXPLORE", "dialog:EXECUTE",
        ],
        "pokemon": [
            "battle:ATTACK", "battle:DEFEND", "battle:SURVIVE",
            "overworld:NAVIGATE", "overworld:EXPLORE", "overworld:COLLECT",
            "menu:SETUP", "menu:OPTIMIZE",
            "dialog:EXPLORE", "dialog:EXECUTE",
        ],
    }

    # Generic seeds for unknown games: temporal phases × common tags.
    _GENERIC_DEFAULT_SEEDS: List[str] = [
        "early:SETUP", "early:EXECUTE", "early:EXPLORE",
        "mid:EXECUTE", "mid:OPTIMIZE", "mid:NAVIGATE",
        "late:EXECUTE", "late:SURVIVE", "late:OPTIMIZE",
    ]

    @staticmethod
    def _seed_skills_from_intentions(
        episode, game_name: str = "generic",
    ) -> List[str]:
        """Extract unique compound skill labels (phase:tag) from an episode,
        augmented with game-appropriate default seeds.

        Uses the phase detector to produce per-step phase labels, then
        combines them with intention tags via ``make_compound_label`` so
        that e.g. early-game MERGE and endgame MERGE become distinct seeds.

        When the episode yields few labels (common in early training when
        the LLM produces monotone intentions like ``[EXECUTE]`` on every
        step), game-specific default seeds are added so the preference
        ranker always has a diverse skill vocabulary to compare.  This
        mirrors the naming scheme in ``labeling/extract_skillbank_gpt54.py``.
        """
        from skill_agents_grpo.boundary_proposal.signal_extractors import (
            parse_intention_tag,
        )
        from skill_agents_grpo.infer_segmentation.phase_detector import (
            detect_phases,
            make_compound_label,
        )

        exps = episode.experiences
        phases = detect_phases(exps, game_name=game_name)
        labels: set = set()
        for exp, phase in zip(exps, phases):
            intent = getattr(exp, "intentions", None)
            if intent:
                tag = parse_intention_tag(intent)
                if tag != "UNKNOWN":
                    labels.add(make_compound_label(phase, tag))

        defaults = SkillBankAgent._GAME_DEFAULT_SEEDS.get(
            game_name, SkillBankAgent._GENERIC_DEFAULT_SEEDS,
        )
        labels.update(defaults)

        return sorted(labels)

    # ── Stage 1+2: Segment one episode ───────────────────────────────

    def segment_episode(
        self,
        episode,
        env_name: Optional[str] = None,
        skill_names: Optional[List[str]] = None,
        *,
        embedder=None,
        surprisal=None,
        extractor_kwargs: Optional[dict] = None,
    ) -> Tuple[Any, list]:
        """Run Stage 1 (boundary proposal) + Stage 2 (segmentation) on one episode.

        Returns (SegmentationResult, list[SubTask_Experience]).
        Side-effect: accumulates segment records for later stages.
        """
        from skill_agents_grpo.infer_segmentation.episode_adapter import (
            infer_and_segment,
            infer_and_segment_offline,
        )
        from skill_agents_grpo.infer_segmentation.config import (
            ContractFeedbackConfig,
            DurationPriorConfig,
            SegmentationConfig,
            NewSkillConfig,
            PreferenceLearningConfig,
            LLMTeacherConfig,
            get_duration_prior_for_game,
        )
        from skill_agents_grpo.boundary_proposal.proposal import ProposalConfig

        cfg = self.config
        _env = env_name or cfg.env_name
        _game = cfg.game_name
        _skill_names = skill_names or list(self.bank.skill_ids)

        # Always merge intention tags from the current episode so the
        # decoder can discover new skill types beyond the existing bank.
        # Without this, once the bank has e.g. 1 skill, the decoder only
        # sees [that_skill, __NEW__] and the penalty makes __NEW__
        # uncompetitive — locking the bank to a single skill forever.
        seeded = self._seed_skills_from_intentions(episode, game_name=_game)
        _skill_names = sorted(set(_skill_names) | set(seeded))

        T = len(episode.experiences)
        duration_cfg = get_duration_prior_for_game(_game, episode_length=T)

        seg_config = SegmentationConfig(
            method=cfg.segmentation_method,
            duration=duration_cfg,
            new_skill=NewSkillConfig(
                enabled=True,
                penalty=cfg.new_skill_penalty,
            ),
            preference=PreferenceLearningConfig(
                num_iterations=cfg.preference_iterations,
                margin_threshold=cfg.margin_threshold,
                max_queries_per_iter=cfg.max_queries_per_iter,
            ),
            llm_teacher=LLMTeacherConfig(
                model=cfg.llm_model,
                max_concurrent_llm_calls=cfg.max_concurrent_llm_calls,
                **({"max_workers": cfg.llm_teacher_max_workers}
                   if cfg.llm_teacher_max_workers is not None else {}),
                **({"max_tokens": cfg.llm_teacher_max_tokens}
                   if cfg.llm_teacher_max_tokens is not None else {}),
            ),
            contract_feedback=ContractFeedbackConfig(
                mode=cfg.contract_feedback_mode,
                strength=cfg.contract_feedback_strength,
            ),
        )
        proposal_config = ProposalConfig(merge_radius=cfg.merge_radius)
        _extractor_kwargs = extractor_kwargs or {"model": cfg.extractor_model}

        # Build compat_fn from the bank when contract feedback is enabled
        _compat_fn = None
        if cfg.contract_feedback_mode != "off" and len(self.bank) > 0:
            _compat_fn = self.bank.compat_fn

        store = self._get_or_create_preference_store()

        _skill_descs: Optional[Dict[str, str]] = None
        _skill_tags: Optional[Dict[str, List[str]]] = None
        _bank_skill_scores: Optional[Dict[str, float]] = None
        if _skill_names and len(self.bank) > 0:
            _skill_descs = {}
            _skill_tags = {}
            _bank_skill_scores = {}
            for sid in _skill_names:
                sk = self.bank.get_skill(sid)
                if sk is None:
                    continue
                parts = []
                if sk.name:
                    parts.append(sk.name)
                if sk.strategic_description:
                    parts.append(sk.strategic_description)
                if parts:
                    _skill_descs[sid] = " — ".join(parts)
                if sk.tags:
                    _skill_tags[sid] = sk.tags
                try:
                    _bank_skill_scores[sid] = sk.compute_skill_score()
                except Exception:
                    _bank_skill_scores[sid] = 0.5

        if _skill_names:
            result, sub_episodes, store = infer_and_segment(
                episode,
                skill_names=_skill_names,
                env_name=_env,
                config=seg_config,
                proposal_config=proposal_config,
                embedder=embedder,
                surprisal=surprisal,
                preference_store=store,
                extractor_kwargs=_extractor_kwargs,
                compat_fn=_compat_fn,
                game_name=_game,
                skill_descriptions=_skill_descs,
                skill_tags=_skill_tags,
                bank_skill_scores=_bank_skill_scores,
            )
            self._preference_store = store
        else:
            result, sub_episodes = infer_and_segment_offline(
                episode,
                skill_names=[],
                env_name=_env,
                config=seg_config,
                proposal_config=proposal_config,
                embedder=embedder,
                surprisal=surprisal,
                extractor_kwargs=_extractor_kwargs,
                compat_fn=_compat_fn,
                game_name=_game,
            )

        ep_id = getattr(episode, "episode_id", None) or ""
        base = getattr(episode, "task", None) or "traj"
        idx = len(self._traj_lengths)
        traj_id = f"{base}__ep{idx}" if ep_id == "" else f"{base}__ep{ep_id}"
        if traj_id in self._observations_by_traj:
            traj_id = f"{traj_id}_{idx}"
        self._cache_trajectory(episode, traj_id, result)

        return result, sub_episodes

    def _cache_trajectory(self, episode, traj_id: str, result) -> None:
        """Store observations and convert segmentation result to SegmentRecord."""
        exps = episode.experiences
        self._observations_by_traj[traj_id] = [
            getattr(e, "summary_state", None) or getattr(e, "state", None)
            for e in exps
        ]
        self._traj_lengths[traj_id] = len(exps)

        segments = result.segments
        for idx, seg in enumerate(segments):
            seg_id = f"{traj_id}_seg{idx:04d}"
            seg_reward = sum(
                getattr(e, "reward", 0.0) or 0.0
                for e in exps[seg.start: seg.end + 1]
            )
            rec = SegmentRecord(
                seg_id=seg_id,
                traj_id=traj_id,
                t_start=seg.start,
                t_end=seg.end,
                skill_label=seg.assigned_skill,
                cumulative_reward=seg_reward,
            )
            if seg.assigned_skill == "__NEW__":
                self._new_pool.append(rec)
                pred_skill = segments[idx - 1].assigned_skill if idx > 0 else None
                succ_skill = segments[idx + 1].assigned_skill if idx < len(segments) - 1 else None
                self._new_pool_mgr.add(rec, predecessor_skill=pred_skill, successor_skill=succ_skill)
            self._all_segments.append(rec)

    # ── Batch ingestion ──────────────────────────────────────────────

    def ingest_episodes(
        self,
        episodes: Sequence,
        env_name: Optional[str] = None,
        skill_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Tuple[Any, list]]:
        """Ingest a batch of episodes through Stage 1+2, then run Stage 3.

        Returns list of (SegmentationResult, SubTask_Experience list) per episode.
        """
        results = []
        for ep in episodes:
            r = self.segment_episode(ep, env_name=env_name, skill_names=skill_names, **kwargs)
            results.append(r)

        if self._all_segments:
            self.run_contract_learning()

        self._invalidate_query_engine()
        return results

    # ── Predicate extraction for Stage 3 ────────────────────────────

    _KV_NOISE_KEYS = frozenset({
        "step", "step_number", "n/a", "filtered_screen_text",
        "selection_box_text", "expansion_direction",
    })

    @staticmethod
    def _parse_kv_state(obs: Any) -> Dict[str, str]:
        """Parse a state string into a dict, handling ``key=val``,
        ``key: val``, and ``[key]: val`` formats separated by ``|``."""
        if obs is None:
            return {}
        text = obs if isinstance(obs, str) else str(obs)
        result: Dict[str, str] = {}
        for part in text.split("|"):
            part = part.strip()
            if not part:
                continue
            # Try key=value first
            if "=" in part:
                k, _, v = part.partition("=")
            elif ":" in part:
                k, _, v = part.partition(":")
            else:
                continue
            k = k.strip().strip("[]").lower().replace(" ", "_")
            v = v.strip().strip(",")
            if not k or not v or k in SkillBankAgent._KV_NOISE_KEYS:
                continue
            if v.lower() in ("n/a", "none", "-"):
                continue
            # Extract sub-fields from complex values like "RedsHouse2f, (x_max...)"
            if k == "map_name" and "," in v:
                v = v.split(",")[0].strip()
            if k == "your_position_(x,_y)":
                k = "position"
            result[k] = v
        return result

    _PER_STEP_NOISE = frozenset({
        "step", "position", "your_position_(x,_y)",
        "current_money", "reward",
    })

    @staticmethod
    def _kv_to_predicates(kv: Dict[str, str], prev_kv: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Convert a parsed key-value state into namespaced float predicates.

        Also detects changes from *prev_kv* and emits ``event.`` predicates
        for keys that don't change every single step.
        """
        preds: Dict[str, float] = {}
        for k, v in kv.items():
            if k in SkillBankAgent._PER_STEP_NOISE:
                continue
            v_lower = v.lower()
            if v_lower in ("true", "yes", "1"):
                preds[f"world.{k}"] = 1.0
            elif v_lower in ("false", "no", "0", "not in battle",
                              "no more pokemons"):
                preds[f"world.{k}"] = 0.0
            else:
                preds[f"world.{k}={v}"] = 1.0

        if prev_kv is not None:
            for k in set(kv) | set(prev_kv):
                if k in SkillBankAgent._PER_STEP_NOISE:
                    continue
                old_v = prev_kv.get(k)
                new_v = kv.get(k)
                if old_v != new_v and old_v is not None and new_v is not None:
                    preds[f"event.{k}_changed"] = 1.0
                elif old_v is None and new_v is not None:
                    preds[f"event.{k}_appeared"] = 1.0
                elif old_v is not None and new_v is None:
                    preds[f"event.{k}_disappeared"] = 1.0
        return preds

    def _ensure_predicates_extracted(self) -> None:
        """Extract predicates for trajectories not yet processed.

        Uses a two-tier strategy:
        1. **Rule-based** (fast, reliable): parses ``key=value`` state strings
           directly and detects changes between consecutive steps.
        2. **LLM-based** (fallback): calls the LLM only when rule-based
           extraction yields too few predicates.
        """
        traj_ids_needed = [
            tid for tid in self._observations_by_traj
            if tid not in self._predicates_by_traj
        ]
        if not traj_ids_needed:
            return

        for traj_id in traj_ids_needed:
            observations = self._observations_by_traj.get(traj_id, [])
            if not observations:
                self._predicates_by_traj[traj_id] = []
                continue

            kv_states = [self._parse_kv_state(obs) for obs in observations]

            all_preds: List[Dict[str, float]] = []
            prev_kv: Optional[Dict[str, str]] = None
            for kv in kv_states:
                preds = self._kv_to_predicates(kv, prev_kv)
                all_preds.append(preds)
                prev_kv = kv

            n_unique = len({k for p in all_preds for k in p})
            if n_unique < 3:
                llm_preds = self._extract_predicates_llm(traj_id, observations)
                if llm_preds:
                    for i, lp in enumerate(llm_preds):
                        if i < len(all_preds):
                            merged = dict(all_preds[i])
                            merged.update(lp)
                            all_preds[i] = merged
                        else:
                            all_preds.append(lp)

            self._predicates_by_traj[traj_id] = all_preds
            logger.info(
                "Extracted predicates for traj %s: %d timesteps, %d unique predicates",
                traj_id, len(all_preds),
                len({k for p in all_preds for k in p}),
            )

    def _extract_predicates_llm(
        self,
        traj_id: str,
        observations: list,
    ) -> Optional[List[Dict[str, float]]]:
        """LLM-based predicate extraction fallback."""
        model = self.config.llm_model or self.config.extractor_model
        if model is None:
            return None

        try:
            from API_func import ask_model as _raw_ask
        except ImportError:
            return None

        from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models
        from skill_agents_grpo.boundary_proposal.llm_extractor import (
            _PREDICATE_EXTRACTION_PROMPT,
            _parse_json_array,
            _normalize_predicate_keys,
        )

        ask = wrap_ask_for_reasoning_models(_raw_ask, model_hint=model)
        chunk_size = 30
        max_chars = 500

        state_texts = []
        for obs in observations:
            if isinstance(obs, str):
                t = obs[:max_chars] + "..." if len(obs) > max_chars else obs
            elif isinstance(obs, dict):
                t = json.dumps(obs, default=str, ensure_ascii=False)
                if len(t) > max_chars:
                    t = t[:max_chars] + "..."
            elif obs is not None:
                t = str(obs)[:max_chars]
            else:
                t = "(empty)"
            state_texts.append(t)

        all_preds: List[dict] = []
        for start in range(0, len(state_texts), chunk_size):
            end = min(start + chunk_size, len(state_texts))
            chunk = state_texts[start:end]

            states_block = "\n".join(
                f"  t={start + i}: {s}" for i, s in enumerate(chunk)
            )
            prompt = _PREDICATE_EXTRACTION_PROMPT.format(
                states_block=states_block,
                num_states=len(chunk),
            )
            try:
                import time as _time
                from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord

                t0 = _time.time()
                response = ask(prompt, model=model, temperature=0.2, max_tokens=3000)
                elapsed = _time.time() - t0
                parsed = _parse_json_array(response)

                record_io(ColdStartRecord(
                    module="pipeline",
                    function="predicate_extraction",
                    prompt=prompt,
                    response=response or "",
                    parsed={"n_predicates": len(parsed)} if parsed else None,
                    model=model or "",
                    temperature=0.2,
                    max_tokens=3000,
                    elapsed_s=round(elapsed, 3),
                    segment_start=start,
                    segment_end=end,
                    n_steps=len(chunk),
                    error=None if parsed else "parse_failed",
                ))

                if parsed is not None:
                    chunk_preds = [p if isinstance(p, dict) else {} for p in parsed]
                    while len(chunk_preds) < len(chunk):
                        chunk_preds.append({})
                    chunk_preds = chunk_preds[:len(chunk)]
                else:
                    chunk_preds = [{} for _ in chunk]
            except Exception:
                chunk_preds = [{} for _ in chunk]
            all_preds.extend(chunk_preds)

        all_preds = _normalize_predicate_keys(all_preds)
        return [self._convert_to_float_predicates(p) for p in all_preds]

    @staticmethod
    def _convert_to_float_predicates(pred_dict: dict) -> Dict[str, float]:
        """Convert LLM predicate dict (mixed types) to Stage 3 float predicates.

        String values become ``key=value`` with prob 1.0 so that changes
        between e.g. ``phase=opening`` and ``phase=midgame`` are captured
        as eff_add / eff_del.
        """
        result: Dict[str, float] = {}
        for k, v in pred_dict.items():
            if v is None:
                continue
            if isinstance(v, bool):
                result[k] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                if 0 <= v <= 1:
                    result[k] = float(v)
                else:
                    result[f"{k}={v}"] = 1.0
            elif isinstance(v, str):
                result[f"{k}={v}"] = 1.0
            else:
                result[k] = 1.0
        return result

    # ── Stage 3: contract learning ───────────────────────────────────

    def run_contract_learning(self) -> Any:
        """Run Stage 3 MVP: learn, verify, and refine contracts for all accumulated segments.

        Returns Stage3MVPSummary.
        """
        from skill_agents_grpo.stage3_mvp.run_stage3_mvp import (
            run_stage3_mvp,
            SegmentSpec,
            Stage3MVPSummary,
        )
        from skill_agents_grpo.stage3_mvp.config import Stage3MVPConfig

        cfg = self.config
        s3_config = Stage3MVPConfig(
            eff_freq=cfg.eff_freq,
            min_instances_per_skill=cfg.min_instances_per_skill,
            start_end_window=cfg.start_end_window,
            model=cfg.llm_model or "",
        )

        specs = [
            SegmentSpec(
                seg_id=rec.seg_id,
                traj_id=rec.traj_id,
                t_start=rec.t_start,
                t_end=rec.t_end,
                skill_label=rec.skill_label,
                ui_events=list(rec.events) if rec.events else [],
            )
            for rec in self._all_segments
            if rec.skill_label.upper() != "NEW" and rec.skill_label != "__NEW__"
        ]

        if not specs:
            logger.info("No non-NEW segments to process in Stage 3.")
            return Stage3MVPSummary()

        self._ensure_predicates_extracted()

        summary = run_stage3_mvp(
            segments=specs,
            observations_by_traj=self._observations_by_traj,
            config=s3_config,
            bank=self.bank,
            bank_path=cfg.bank_path,
            precomputed_predicates_by_traj=self._predicates_by_traj or None,
        )
        logger.info("Stage 3: %s", summary)
        self._invalidate_query_engine()
        return summary

    # ── Protocol update ────────────────────────────────────────────

    def update_protocols(self) -> int:
        """Synthesize or update protocols for skills that need it.

        Called after quality check identifies skills with enough high-quality
        sub-episodes.  Uses LLM to synthesize actionable step-by-step
        protocols from sub-episode evidence.

        Returns the number of protocols updated.
        """
        from skill_agents_grpo.stage3_mvp.schemas import Protocol, Skill

        updated = 0
        from_template = 0
        for sid in self.bank.skill_ids:
            skill = self.bank.get_skill(sid)
            if skill is None:
                continue

            has_steps = bool(skill.protocol and skill.protocol.steps)
            proto_source = getattr(skill.protocol, "source", "template") if skill.protocol else "template"
            has_llm_protocol = has_steps and proto_source == "llm"

            q_threshold = 0.6 if has_llm_protocol else 0.35
            high_quality = [se for se in skill.sub_episodes if se.quality_score >= q_threshold]

            # LLM protocols need strong evidence (3+) before re-synthesis.
            # Template / deterministic protocols are eagerly upgraded with
            # just 1 sub-episode of evidence.
            if has_llm_protocol and len(high_quality) < 3:
                continue
            if not has_llm_protocol and len(high_quality) < 1:
                continue

            old_source = proto_source
            protocol = self._synthesize_protocol(skill, high_quality)
            if protocol is not None:
                skill.bump_version()
                skill.protocol = protocol
                self.bank.add_or_update_skill(skill)
                updated += 1
                if old_source == "template":
                    from_template += 1
                logger.info(
                    "Updated protocol for skill %s (v%d, %d steps, %s->%s, %d evidence)",
                    sid, skill.version, len(protocol.steps),
                    old_source, protocol.source, len(high_quality),
                )

        if updated:
            self._invalidate_query_engine()
            if from_template:
                logger.info(
                    "Protocol synthesis: %d/%d upgraded from template to LLM",
                    from_template, updated,
                )
        return updated

    def refine_low_pass_protocols(
        self,
        skill_score_threshold: float = 0.35,
        min_episodes: int = 3,
    ) -> int:
        """Re-synthesize protocols for skills with a low skill_score.

        Called periodically (e.g. every N co-evolution iterations) to give
        struggling skills a chance to improve their protocols using the
        latest sub-episode evidence.

        Only re-synthesizes if the skill already has a protocol and at
        least *min_episodes* sub-episodes.  The new protocol replaces the
        old one.

        Returns the number of protocols re-synthesized.
        """
        from skill_agents_grpo.stage3_mvp.schemas import Skill

        refined = 0
        for sid in self.bank.skill_ids:
            skill = self.bank.get_skill(sid)
            if skill is None:
                continue
            if not skill.protocol or not skill.protocol.steps:
                continue
            if skill.compute_skill_score() >= skill_score_threshold:
                continue
            if len(skill.sub_episodes) < min_episodes:
                continue

            all_eps = sorted(
                skill.sub_episodes, key=lambda se: se.quality_score, reverse=True,
            )
            best_eps = all_eps[:max(3, len(all_eps) // 2)]

            old_source = getattr(skill.protocol, "source", "template")
            new_protocol = self._synthesize_protocol(skill, best_eps)
            if new_protocol is not None:
                skill.bump_version()
                skill.protocol = new_protocol
                self.bank.add_or_update_skill(skill)
                refined += 1
                logger.info(
                    "Refined protocol for low-pass skill %s "
                    "(score=%.2f, v%d, %s->%s, %d evidence)",
                    sid, skill.compute_skill_score(), skill.version,
                    old_source, new_protocol.source, len(best_eps),
                )

        if refined:
            self._invalidate_query_engine()
        return refined

    def _synthesize_protocol(
        self,
        skill: Skill,
        high_quality_eps: list,
    ) -> Optional:
        """Synthesize a Protocol from high-quality sub-episode pointers.

        Uses sub-episode summaries and intention_tags (cached on the
        pointers) plus contract effects.  When an LLM model is configured
        (via ``config.llm_model`` or ``config.extractor_model``), generates
        richer protocols via the same ``ask_model`` routing used by both
        GPT and Qwen backends.
        """
        from skill_agents_grpo.stage3_mvp.schemas import Protocol

        contract = skill.contract

        # Derive expected_tag_pattern from the majority tag set
        from collections import Counter
        tag_counter: Counter = Counter()
        for se in high_quality_eps:
            tag_counter.update(se.intention_tags)
        if tag_counter:
            skill.expected_tag_pattern = [
                tag for tag, _ in tag_counter.most_common(5)
            ]

        avg_len = 0
        if high_quality_eps:
            lengths = [se.length for se in high_quality_eps]
            from decision_agents.protocol_utils import compute_expected_duration
            avg_len = compute_expected_duration(lengths)

        # Collect evidence for LLM synthesis
        summaries = [se.summary for se in high_quality_eps if se.summary]
        effects_desc = ""
        if contract:
            parts = []
            if contract.eff_add:
                parts.append("achieves: " + ", ".join(sorted(contract.eff_add)[:5]))
            if contract.eff_del:
                parts.append("removes: " + ", ".join(sorted(contract.eff_del)[:3]))
            if contract.eff_event:
                parts.append("triggers: " + ", ".join(sorted(contract.eff_event)[:3]))
            effects_desc = "; ".join(parts)

        # Try LLM-based synthesis (model-agnostic: routes to GPT or Qwen)
        llm_model = self.config.llm_model or self.config.extractor_model
        action_vocab = getattr(self.config, "action_vocab", None) or []
        protocol = self._llm_synthesize_protocol(
            skill, summaries, effects_desc, llm_model,
            action_vocab=action_vocab,
        )
        if protocol is not None:
            protocol.expected_duration = max(1, avg_len)
            if skill.success_rate < 0.5:
                protocol.abort_criteria.append(
                    "Abort if no progress after expected duration"
                )
            return protocol

        # Deterministic fallback (no LLM available)
        steps = []
        preconditions = []
        success_criteria = []

        if summaries:
            seen = set()
            for s in summaries:
                key = s.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    steps.append(s.strip())
                if len(steps) >= 5:
                    break

        if not steps and contract:
            if contract.eff_add:
                for lit in sorted(contract.eff_add)[:5]:
                    steps.append(f"Achieve: {lit}")
                    success_criteria.append(f"{lit} is true")
            if contract.eff_del:
                for lit in sorted(contract.eff_del)[:3]:
                    steps.append(f"Remove: {lit}")
            if contract.eff_event:
                for lit in sorted(contract.eff_event)[:3]:
                    steps.append(f"Trigger: {lit}")

        if not steps:
            steps = ["Execute skill actions as needed"]

        abort_criteria = []
        if skill.success_rate < 0.5:
            abort_criteria.append("Abort if no progress after expected duration")

        return Protocol(
            preconditions=preconditions,
            steps=steps,
            success_criteria=success_criteria,
            abort_criteria=abort_criteria,
            expected_duration=max(1, avg_len),
            source="deterministic",
        )

    def _llm_synthesize_protocol(
        self,
        skill,
        summaries: List[str],
        effects_desc: str,
        model: Optional[str],
        action_vocab: Optional[List[str]] = None,
    ) -> Optional:
        """Generate a structured protocol via LLM.

        Tries local vLLM (Qwen) first with retry, then falls back to
        ``ask_model`` (OpenRouter/GPT).  Sets ``protocol.source`` to
        ``"llm"`` on success so tag-based enrichment won't overwrite it.
        """
        from skill_agents_grpo.stage3_mvp.schemas import Protocol
        import json as _json

        evidence = "\n".join(f"  - {s}" for s in summaries[:5]) if summaries else "(none)"
        skill_name = skill.name or skill.skill_id

        action_block = ""
        if action_vocab:
            action_block = (
                f"\nAvailable game actions: {', '.join(action_vocab[:20])}\n"
                f"IMPORTANT: Reference these exact action names in your steps.\n"
            )

        prompt = (
            f"You are a game-AI protocol designer. Generate a concrete execution "
            f"protocol for the skill below.\n\n"
            f"Skill: {skill_name}\n"
            f"Description: {skill.strategic_description or '(none)'}\n"
            f"Effects: {effects_desc or '(none)'}\n"
            f"Evidence from successful executions:\n{evidence}\n"
            f"{action_block}\n"
            f"Generate a structured protocol as JSON with these exact keys:\n"
            f'{{"preconditions": ["..."], "steps": ["..."], '
            f'"step_checks": ["..."], '
            f'"success_criteria": ["..."], "abort_criteria": ["..."], '
            f'"predicate_success": ["key=value", ...], '
            f'"predicate_abort": ["key>N", ...]}}\n'
            f"Rules:\n"
            f"- preconditions: 1-3 specific conditions (game situation, state) "
            f"that must hold before starting\n"
            f"- steps: 2-7 concrete action steps (imperative, game-specific). "
            f"Do NOT write generic steps like 'Achieve: X' or 'Execute skill'. "
            f"Reference actual game actions when possible.\n"
            f"- step_checks: one entry per step — a key=value condition from "
            f"the game state that indicates the step is complete "
            f"(e.g. 'stack_h<5', 'quest=2'). Use empty string '' if no "
            f"specific check applies.\n"
            f"- success_criteria: 1-3 human-readable descriptions of success\n"
            f"- abort_criteria: 1-2 human-readable conditions to stop early\n"
            f"- predicate_success: 1-3 machine-checkable key=value or key<N "
            f"conditions from the game state (e.g. 'phase=endgame', "
            f"'holes<5')\n"
            f"- predicate_abort: 1-2 machine-checkable conditions "
            f"(e.g. 'stack_h>18', 'moves<3')\n"
            f"Reply with ONLY the JSON object."
        )

        reply = self._ask_protocol_llm(prompt, model)
        if not reply:
            return None

        try:
            import time as _time_mod
            from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord

            t0 = _time_mod.time()

            import re
            json_m = re.search(r"\{[\s\S]*\}", reply)
            if not json_m:
                record_io(ColdStartRecord(
                    module="pipeline",
                    function="protocol_synthesis",
                    prompt=prompt,
                    response=reply,
                    model=model or "vllm",
                    temperature=0.3,
                    max_tokens=800,
                    elapsed_s=0.0,
                    skill_id=skill.skill_id,
                    error="no_json_found",
                ))
                return None
            data = _json.loads(json_m.group(0))

            record_io(ColdStartRecord(
                module="pipeline",
                function="protocol_synthesis",
                prompt=prompt,
                response=reply,
                parsed=data,
                model=model or "vllm",
                temperature=0.3,
                max_tokens=800,
                elapsed_s=0.0,
                skill_id=skill.skill_id,
            ))

            steps = data.get("steps", [])[:7]
            step_checks = data.get("step_checks", [])[:7]
            if step_checks and len(step_checks) < len(steps):
                step_checks.extend([""] * (len(steps) - len(step_checks)))
            return Protocol(
                preconditions=data.get("preconditions", [])[:5],
                steps=steps,
                success_criteria=data.get("success_criteria", [])[:5],
                abort_criteria=data.get("abort_criteria", [])[:3],
                step_checks=step_checks,
                predicate_success=data.get("predicate_success", [])[:5],
                predicate_abort=data.get("predicate_abort", [])[:3],
                action_vocab=action_vocab or [],
                source="llm",
            )
        except Exception as exc:
            logger.warning("LLM protocol synthesis parse failed: %s", exc)
            return None

    @staticmethod
    def _ask_protocol_llm(prompt: str, model: Optional[str]) -> str:
        """Try local vLLM with retry, fall back to ask_model."""
        try:
            from skill_agents_grpo._llm_retry import sync_ask_with_retry
            from API_func import ask_vllm
            reply = sync_ask_with_retry(
                ask_vllm,
                prompt,
                log_label="protocol_synth_vllm",
                temperature=0.3,
                max_tokens=800,
            )
            if reply and not reply.startswith("Error"):
                return reply
        except Exception as exc:
            logger.debug("vLLM protocol call failed, trying ask_model: %s", exc)

        if model is None:
            return ""
        try:
            from API_func import ask_model as _raw_ask
            from skill_agents_grpo._llm_compat import wrap_ask_for_reasoning_models
            _ask = wrap_ask_for_reasoning_models(_raw_ask, model_hint=model)
            reply = _ask(prompt, model=model, temperature=0.3, max_tokens=800)
            if reply and not reply.startswith("Error"):
                return reply
        except Exception as exc:
            logger.warning("ask_model protocol fallback failed: %s", exc)
        return ""

    # ── Phase 5: Distill execution hints ────────────────────────────

    def distill_execution_hints(self, min_successful: int = 3) -> int:
        """Derive lightweight execution hints for skills with enough evidence.

        For each skill, aggregates successful sub-episodes to produce:
        - common preconditions
        - common target objects
        - state-transition pattern
        - termination cues
        - common failure modes
        - natural-language execution description

        Returns the number of skills updated with hints.
        """
        from skill_agents_grpo.stage3_mvp.schemas import ExecutionHint
        from collections import Counter

        updated = 0
        for sid in self.bank.skill_ids:
            skill = self.bank.get_skill(sid)
            if skill is None:
                continue

            successful = [
                se for se in skill.sub_episodes
                if se.outcome == "success" or se.quality_score >= 0.6
            ]
            if len(successful) < min_successful:
                continue

            # Skip if hint is recent and based on enough data
            if (skill.execution_hint is not None
                    and skill.execution_hint.n_source_segments >= len(successful)):
                continue

            contract = skill.contract

            # --- common preconditions ---
            preconditions: List[str] = []
            if skill.protocol.preconditions:
                preconditions = list(skill.protocol.preconditions[:5])

            # --- common target objects ---
            obj_counter: Counter = Counter()
            for se in successful:
                for tag in se.intention_tags:
                    parts = tag.strip("[]").split("_")
                    for p in parts:
                        if p and len(p) > 2 and p.upper() != p:
                            obj_counter[p.lower()] += 1
            target_objects = [obj for obj, _ in obj_counter.most_common(5)]

            # --- state-transition pattern ---
            transition_pattern = ""
            if contract is not None:
                add_str = ", ".join(sorted(contract.eff_add)[:3]) if contract.eff_add else ""
                del_str = ", ".join(sorted(contract.eff_del)[:3]) if contract.eff_del else ""
                parts = []
                if del_str:
                    parts.append(f"removes [{del_str}]")
                if add_str:
                    parts.append(f"achieves [{add_str}]")
                transition_pattern = " → ".join(parts)

            # --- termination cues ---
            termination_cues: List[str] = []
            if skill.protocol.success_criteria:
                termination_cues = list(skill.protocol.success_criteria[:3])
            elif contract is not None and contract.eff_add:
                termination_cues = [
                    f"{lit} becomes true" for lit in sorted(contract.eff_add)[:3]
                ]

            # --- failure modes ---
            failure_modes: List[str] = []
            report = self.bank.get_report(sid)
            if report is not None and report.failure_signatures:
                failure_modes = [
                    f"{sig} ({cnt}x)"
                    for sig, cnt in sorted(
                        report.failure_signatures.items(), key=lambda x: -x[1]
                    )[:3]
                ]
            if skill.protocol.abort_criteria:
                failure_modes.extend(skill.protocol.abort_criteria[:2])

            if not failure_modes and skill.tags:
                primary_tag = skill.tags[0].upper() if skill.tags else ""
                _TAG_FAILURE_DEFAULTS = {
                    "SURVIVE": "Board state deteriorates despite defensive moves",
                    "DEFEND": "Board state deteriorates despite defensive moves",
                    "MERGE": "No merge opportunities available on any legal move",
                    "POSITION": "Structure broken — anchor tile dislodged or ordering disrupted",
                    "SETUP": "Structure broken — anchor tile dislodged or ordering disrupted",
                    "CLEAR": "Clearing move creates worse congestion than before",
                    "ATTACK": "Attack creates vulnerability without achieving objective",
                    "NAVIGATE": "Path blocked or position unchanged after several moves",
                    "COLLECT": "Target resource depleted or unreachable",
                    "BUILD": "Required materials unavailable or placement blocked",
                }
                if primary_tag in _TAG_FAILURE_DEFAULTS:
                    failure_modes.append(_TAG_FAILURE_DEFAULTS[primary_tag])
                else:
                    failure_modes.append(
                        "No progress toward skill objective after several moves"
                    )

            # --- execution description ---
            exec_desc = ""
            if skill.strategic_description:
                exec_desc = skill.strategic_description
            elif skill.protocol.steps:
                exec_desc = " → ".join(skill.protocol.steps[:4])
            elif successful:
                summaries = [se.summary for se in successful if se.summary][:3]
                if summaries:
                    exec_desc = "; ".join(summaries)

            # Compute typical durations
            lengths = [se.length for se in successful]
            avg_len = sum(lengths) / max(len(lengths), 1)

            hint = ExecutionHint(
                common_preconditions=preconditions,
                common_target_objects=target_objects,
                state_transition_pattern=transition_pattern,
                termination_cues=termination_cues,
                common_failure_modes=failure_modes[:5],
                execution_description=exec_desc,
                n_source_segments=len(successful),
            )

            skill.execution_hint = hint
            self.bank.add_or_update_skill(skill)
            updated += 1

        if updated:
            logger.info("Distilled execution hints for %d skills.", updated)
            self._invalidate_query_engine()

        return updated

    # ── Stage 4.5: sub-episode quality check ────────────────────────

    def run_sub_episode_quality_check(self) -> List[dict]:
        """Run quality scoring, drop low-quality sub-episodes, flag
        skills needing protocol updates, and retire depleted skills.

        Returns list of per-skill quality check results.
        """
        from skill_agents_grpo.quality.sub_episode_evaluator import run_quality_check_batch

        skills = [
            self.bank.get_skill(sid)
            for sid in self.bank.skill_ids
            if self.bank.get_skill(sid) is not None
        ]
        results = run_quality_check_batch(skills, bank=self.bank)

        for r in results:
            if r.get("retired"):
                logger.info("Retiring skill %s after quality check.", r["skill_id"])
            self.bank.recompute_stats(r["skill_id"])

        if results:
            self._invalidate_query_engine()
        return results

    # ── Stage 4: bank maintenance ────────────────────────────────────

    def run_bank_maintenance(
        self,
        embeddings: Optional[Dict[str, List[float]]] = None,
        stage2_diagnostics: Optional[List[dict]] = None,
    ) -> Any:
        """Run Stage 4 core: split, merge, refine skills and local re-decode.

        Returns BankMaintenanceResult.

        Note: materialize and promote are kept as separate calls so callers
        (e.g. the extraction script) can checkpoint between each operation.
        Use ``run_stage4_full`` to execute all five operations in one call.
        """
        from skill_agents_grpo.bank_maintenance.run_bank_maintenance import (
            run_bank_maintenance,
            BankMaintenanceResult,
        )
        from skill_agents_grpo.bank_maintenance.config import BankMaintenanceConfig

        cfg = self.config

        maint_config = BankMaintenanceConfig(
            split_pass_rate_thresh=cfg.split_pass_rate_threshold,
            child_pass_rate_thresh=cfg.child_pass_rate_threshold,
            merge_eff_jaccard_thresh=cfg.merge_jaccard_threshold,
            merge_emb_cosine_thresh=cfg.merge_embedding_threshold,
            min_child_size=cfg.min_child_size,
        )

        report_path = None
        if cfg.report_dir:
            report_path = str(Path(cfg.report_dir) / f"bank_diff_iter{self._iteration}.json")

        result = run_bank_maintenance(
            bank=self.bank,
            all_segments=self._all_segments,
            config=maint_config,
            embeddings=embeddings,
            stage2_diagnostics=stage2_diagnostics,
            traj_lengths=self._traj_lengths,
            report_path=report_path,
        )

        if result.alias_map:
            self._apply_alias_map(result.alias_map)

        self._invalidate_query_engine()
        logger.info(
            "Stage 4: %d splits, %d merges, %d refines",
            len(result.split_results),
            len(result.merge_results),
            len(result.refine_results),
        )
        return result

    def _apply_alias_map(self, alias_map: Dict[str, str]) -> None:
        """Relabel segment records after merges."""
        for seg in self._all_segments:
            if seg.skill_label in alias_map:
                seg.skill_label = alias_map[seg.skill_label]
        for seg in self._new_pool:
            if seg.skill_label in alias_map:
                seg.skill_label = alias_map[seg.skill_label]

    # ── Proto-Skill Management ────────────────────────────────────────

    def form_proto_skills(self) -> int:
        """Scan NEW pool clusters and form proto-skills.

        Proto-skills are lightweight intermediates that can participate
        in Stage 2 decoding before full promotion.

        Returns the number of proto-skills created.
        """
        existing = set(self.bank.skill_ids)
        created = self._proto_mgr.form_from_pool(
            self._new_pool_mgr, existing_bank_skills=existing,
        )
        if created:
            logger.info(
                "Formed %d proto-skills: %s",
                len(created),
                [p.proto_id for p in created[:5]],
            )
        return len(created)

    def verify_proto_skills(self) -> int:
        """Run light verification on all unverified proto-skills.

        Returns the number of proto-skills verified.
        """
        verified = 0
        for pid in list(self._proto_mgr.proto_ids):
            proto = self._proto_mgr.get(pid)
            if proto is None or proto.verified:
                continue
            pass_rate = self._proto_mgr.verify(
                pid, self.bank, self._observations_by_traj,
            )
            if pass_rate is not None:
                verified += 1
                logger.info(
                    "Proto-skill %s verified: pass_rate=%.3f",
                    pid, pass_rate,
                )
        return verified

    def promote_proto_skills(self) -> List[str]:
        """Promote qualifying proto-skills to real skills.

        Returns the list of promoted skill IDs.
        """
        promoted = self._proto_mgr.promote_ready(self.bank)
        if promoted:
            logger.info("Promoted %d proto-skills to real skills: %s", len(promoted), promoted[:5])
            self._invalidate_query_engine()
            if self.config.bank_path:
                self.bank.save(self.config.bank_path)
        return promoted or []

    # ── Materialize NEW ──────────────────────────────────────────────

    def materialize_new_skills(self) -> List[str]:
        """Promote qualifying ``__NEW__`` clusters to real skills.

        Uses the ``NewPoolManager`` for rich clustering (effect similarity,
        consistency, separability).  Falls back to the legacy signature-based
        approach only when the pool manager is empty but _new_pool has records.

        Returns the list of newly created skill IDs.
        """
        from skill_agents_grpo.infer_segmentation.config import LLMTeacherConfig

        _mw = self.config.llm_teacher_max_workers
        _mt = self.config.llm_teacher_max_tokens
        llm_cfg = LLMTeacherConfig(
            model=self.config.llm_model,
            max_concurrent_llm_calls=self.config.max_concurrent_llm_calls,
            **({"max_workers": _mw} if _mw is not None else {}),
            **({"max_tokens": _mt} if _mt is not None else {}),
        ) if self.config.llm_model else None

        # Try the new pool manager first
        if self._new_pool_mgr.size >= self.config.new_pool_min_cluster_size:
            logger.info(
                "NEW pool: %d candidates, running promotion.",
                self._new_pool_mgr.size,
            )
            created_ids = self._new_pool_mgr.promote(
                bank=self.bank,
                observations_by_traj=self._observations_by_traj,
                llm_config=llm_cfg,
            )
            # Sync legacy pool
            promoted_seg_ids = {c.seg_id for c in self._new_pool_mgr.records}
            self._new_pool = [
                r for r in self._new_pool
                if r.seg_id not in self._new_pool_mgr._promoted_ids
            ]

            for cid in created_ids:
                logger.info("Materialized new skill %s via NewPoolManager.", cid)

            if created_ids and self.config.bank_path:
                self.bank.save(self.config.bank_path)
            self._invalidate_query_engine()
            return created_ids or []

        if len(self._new_pool) < self.config.min_new_cluster_size:
            logger.info(
                "NEW pool too small (%d < %d), skipping materialization.",
                len(self._new_pool), self.config.min_new_cluster_size,
            )
            return []

        logger.info(
            "NEW pool: %d legacy records, running legacy promotion.",
            len(self._new_pool),
        )
        return self._materialize_legacy()

    def _materialize_legacy(self) -> List[str]:
        """Legacy promotion: cluster by exact effect_signature string."""
        from skill_agents_grpo.stage3_mvp.run_stage3_mvp import (
            run_stage3_mvp,
            SegmentSpec,
        )
        from skill_agents_grpo.stage3_mvp.config import Stage3MVPConfig

        by_sig: Dict[str, List[SegmentRecord]] = defaultdict(list)
        for rec in self._new_pool:
            sig = rec.effect_signature()
            by_sig[sig].append(rec)

        created_ids: List[str] = []
        ts = int(time.time())

        for sig, cluster in by_sig.items():
            if len(cluster) < self.config.min_new_cluster_size:
                continue

            new_id = f"S_new_{ts}_{len(created_ids)}"
            for rec in cluster:
                rec.skill_label = new_id

            specs = [
                SegmentSpec(
                    seg_id=rec.seg_id,
                    traj_id=rec.traj_id,
                    t_start=rec.t_start,
                    t_end=rec.t_end,
                    skill_label=new_id,
                    ui_events=list(rec.events) if getattr(rec, "events", None) else [],
                )
                for rec in cluster
            ]

            s3_config = Stage3MVPConfig(
                eff_freq=self.config.eff_freq,
                min_instances_per_skill=max(1, self.config.min_new_cluster_size),
            )

            summary = run_stage3_mvp(
                segments=specs,
                observations_by_traj=self._observations_by_traj,
                config=s3_config,
                bank=self.bank,
            )

            if new_id in summary.skill_results:
                sr = summary.skill_results[new_id]
                if sr.get("pass_rate", 0) >= self.config.split_pass_rate_threshold:
                    created_ids.append(new_id)
                    for rec in cluster:
                        if rec in self._new_pool:
                            self._new_pool.remove(rec)
                else:
                    self.bank.remove(new_id)
                    for rec in cluster:
                        rec.skill_label = "__NEW__"

        if created_ids and self.config.bank_path:
            self.bank.save(self.config.bank_path)
        self._invalidate_query_engine()
        return created_ids

    # ── Skill evaluation ─────────────────────────────────────────────

    def run_evaluation(
        self,
        episode_outcomes: Optional[Dict[str, bool]] = None,
    ) -> Any:
        """Run the skill evaluation pipeline.

        Returns EvaluationSummary.
        """
        from skill_agents_grpo.skill_evaluation.run_evaluation import run_skill_evaluation
        from skill_agents_grpo.skill_evaluation.config import SkillEvaluationConfig

        eval_config = SkillEvaluationConfig()
        if self.config.llm_model:
            eval_config.llm.model = self.config.llm_model

        report_path = None
        if self.config.report_dir:
            report_path = str(Path(self.config.report_dir) / f"eval_iter{self._iteration}.json")

        summary = run_skill_evaluation(
            bank=self.bank,
            all_segments=self._all_segments,
            config=eval_config,
            episode_outcomes=episode_outcomes,
            report_path=report_path,
        )
        logger.info("Evaluation: %d skills evaluated", len(summary.skill_reports) if hasattr(summary, 'skill_reports') else 0)
        return summary

    # ── Full iteration ───────────────────────────────────────────────

    def run_full_iteration(
        self,
        episodes: Optional[Sequence] = None,
        env_name: Optional[str] = None,
        skill_names: Optional[List[str]] = None,
        **kwargs,
    ) -> IterationSnapshot:
        """Execute one full pipeline iteration.

        Pipeline: (optional ingest) → Stage 3 → quality check
        → Stage 4 (split/merge/refine) → materialize → promote
        → execution hints → snapshot.

        If *episodes* is provided, runs Stage 1+2 first.
        """
        self._iteration += 1
        logger.info("=== Pipeline iteration %d ===", self._iteration)

        if episodes is not None:
            self.ingest_episodes(episodes, env_name=env_name, skill_names=skill_names, **kwargs)
        else:
            if self._all_segments:
                self.run_contract_learning()

        self.run_sub_episode_quality_check()
        self.run_bank_maintenance()

        # Stage 4 continued: materialize + promote
        n_proto = self.form_proto_skills()
        if n_proto > 0:
            self.verify_proto_skills()
        promoted_ids = self.promote_proto_skills()

        materialized_ids = self.materialize_new_skills()

        # Curator pass for materialize/promote actions
        if materialized_ids or promoted_ids:
            self._curator_review_materialize_promote(materialized_ids, promoted_ids)

        # Phase 5: distill execution hints for skills with enough evidence
        self.distill_execution_hints()

        snap = self._take_snapshot(
            n_materialized=len(materialized_ids) + len(promoted_ids),
        )
        self._history.append(snap)
        return snap

    def _curator_review_materialize_promote(
        self,
        materialized_ids: List[str],
        promoted_ids: List[str],
    ) -> None:
        """Run a curator pass on materialize/promote actions.

        Creates synthetic BankMaintenanceResult entries so
        ``_collect_curator_candidates`` includes them, then calls the
        curator filter to generate GRPO training data.
        """
        from skill_agents_grpo.bank_maintenance.run_bank_maintenance import (
            BankMaintenanceResult, _collect_curator_candidates,
        )
        result = BankMaintenanceResult()
        result.materialized_ids = list(materialized_ids)
        result.promoted_ids = list(promoted_ids)

        candidates = _collect_curator_candidates(result, bank=self.bank)
        if not candidates:
            return

        try:
            from skill_agents_grpo.bank_maintenance.llm_curator import filter_candidates
            filter_candidates(candidates, self.bank)
        except Exception as exc:
            logger.debug("Curator filtering for materialize/promote skipped: %s", exc)

    def _take_snapshot(self, n_materialized: int = 0, maint_result=None) -> IterationSnapshot:
        n_new = sum(1 for s in self._all_segments if s.skill_label in ("__NEW__", "NEW"))
        n_total = len(self._all_segments) or 1

        pass_rates = []
        for sid in self.bank.skill_ids:
            r = self.bank.get_report(sid)
            if r is not None:
                pass_rates.append(r.overall_pass_rate)

        return IterationSnapshot(
            iteration=self._iteration,
            n_skills=len(self.bank),
            n_segments=len(self._all_segments),
            n_new_segments=n_new,
            new_rate=n_new / n_total,
            mean_pass_rate=sum(pass_rates) / len(pass_rates) if pass_rates else 0.0,
            n_materialized=n_materialized,
        )

    # ── Convergence loop ─────────────────────────────────────────────

    def run_until_stable(
        self,
        max_iterations: Optional[int] = None,
    ) -> List[IterationSnapshot]:
        """Iterate Stage 3 → Stage 4 until convergence or max iterations.

        Convergence: NEW rate < threshold and merge/split events become rare.
        """
        max_iter = max_iterations or self.config.max_iterations
        snapshots: List[IterationSnapshot] = []

        for _ in range(max_iter):
            snap = self.run_full_iteration()
            snapshots.append(snap)

            if self._is_converged(snap):
                logger.info("Pipeline converged at iteration %d", snap.iteration)
                break

        self.save()
        return snapshots

    def _is_converged(self, snap: IterationSnapshot) -> bool:
        if snap.new_rate > self.config.convergence_new_rate:
            return False
        if snap.n_splits > 0 or snap.n_merges > 0:
            return False
        if snap.mean_pass_rate < self.config.split_pass_rate_threshold:
            return False
        return True

    # ── Skill CRUD ───────────────────────────────────────────────────

    def add_skill(
        self,
        skill_id: str,
        eff_add: Optional[set] = None,
        eff_del: Optional[set] = None,
        eff_event: Optional[set] = None,
    ) -> SkillEffectsContract:
        """Manually add a skill to the bank."""
        contract = SkillEffectsContract(
            skill_id=skill_id,
            eff_add=eff_add or set(),
            eff_del=eff_del or set(),
            eff_event=eff_event or set(),
        )
        self.bank.add_or_update(contract)
        self._invalidate_query_engine()
        logger.info("Added skill %s", skill_id)
        return contract

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill from the bank."""
        if not self.bank.has_skill(skill_id):
            return False
        self.bank.remove(skill_id)
        self._invalidate_query_engine()
        logger.info("Removed skill %s", skill_id)
        return True

    def update_skill(
        self,
        skill_id: str,
        eff_add: Optional[set] = None,
        eff_del: Optional[set] = None,
        eff_event: Optional[set] = None,
    ) -> Optional[SkillEffectsContract]:
        """Update an existing skill's contract.  Returns updated contract or None."""
        contract = self.bank.get_contract(skill_id)
        if contract is None:
            logger.warning("Skill %s not found for update", skill_id)
            return None

        if eff_add is not None:
            contract.eff_add = eff_add
        if eff_del is not None:
            contract.eff_del = eff_del
        if eff_event is not None:
            contract.eff_event = eff_event
        contract.bump_version()
        self.bank.add_or_update(contract)
        self._invalidate_query_engine()
        logger.info("Updated skill %s to version %d", skill_id, contract.version)
        return contract

    # ── Query API (used by decision_agents) ──────────────────────────

    def query_skill(
        self,
        key: str,
        top_k: int = 3,
        current_state: Optional[Dict[str, float]] = None,
        current_predicates: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Query skills by natural-language key (scene/objective/entities).

        When *current_state* or *current_predicates* is provided, defaults
        to the rich state-aware ``select()`` path which uses applicability,
        pass rate, matched/missing effects, and state-conditioned ranking.

        Falls back to the simpler retrieval-only ``query()`` path only when
        no state information is available.

        Returns a list of structured guidance dicts suitable for
        decision_agents.
        """
        engine = self._get_query_engine()
        state = current_state or current_predicates
        if state is not None:
            results = engine.select(
                key, current_state=state, top_k=top_k,
            )
            return [r.to_dict() for r in results]
        return engine.query(key, top_k=top_k)

    def select_skill(
        self,
        query: str,
        current_state: Optional[Dict[str, float]] = None,
        current_predicates: Optional[Dict[str, float]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Rich skill selection combining retrieval relevance with execution
        applicability and structured guidance.

        Preferred API for decision agents.  When ``current_state`` or
        ``current_predicates`` is provided, the result includes full
        guidance: applicability, confidence, why_selected, expected_effects,
        preconditions, termination_hint, failure_modes, execution_hint.

        Returns list of ``SkillSelectionResult.to_dict()``.
        """
        engine = self._get_query_engine()
        state = current_state or current_predicates
        results = engine.select(
            query, current_state=state, top_k=top_k,
        )
        return [r.to_dict() for r in results]

    def query_by_effects(
        self,
        desired_add: Optional[set] = None,
        desired_del: Optional[set] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find skills whose effects best match desired state changes."""
        engine = self._get_query_engine()
        return engine.query_by_effects(desired_add=desired_add, desired_del=desired_del, top_k=top_k)

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all skills with compact summaries."""
        engine = self._get_query_engine()
        return engine.list_all()

    def get_skill_detail(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get full detail for one skill (contract + report + quality)."""
        engine = self._get_query_engine()
        return engine.get_detail(skill_id)

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def skill_ids(self) -> List[str]:
        return self.bank.skill_ids

    def get_contract(self, skill_id: str) -> Optional[SkillEffectsContract]:
        return self.bank.get_contract(skill_id)

    def get_bank(self) -> SkillBankMVP:
        return self.bank

    @property
    def segments(self) -> List[SegmentRecord]:
        return list(self._all_segments)

    @property
    def iteration_history(self) -> List[IterationSnapshot]:
        return list(self._history)
