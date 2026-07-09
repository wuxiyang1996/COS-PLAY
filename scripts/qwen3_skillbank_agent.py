#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-8B Skill Bank Agent (GRPO pipeline) — extract skills from GPT-5.4
rollouts using the **skill_agents_grpo** pipeline with Qwen3-8B as the
LLM backend (served via vLLM).

Mirrors ``extract_skillbank_grpo_gpt54.py`` but replaces the GPT-5.4
backend with a locally-hosted Qwen3-8B, and optionally trains per-stage
LoRA adapters via GRPO.

Pipeline stages (all via ``skill_agents_grpo``):

  Stage 1+2 — Boundary proposal + skill-sequence decoding
              (``SkillBankAgent.segment_episode``)
  Stage 3   — Effects-only contract learning / verify / refine
              (``SkillBankAgent.run_contract_learning``)
  Stage 4.5 — Sub-episode quality check
              (``SkillBankAgent.run_sub_episode_quality_check``)
  Stage 4   — Bank maintenance: split, merge, refine
              (``SkillBankAgent.run_bank_maintenance``)
  Proto     — Proto-skill formation / verification / promotion
              (``form_proto_skills``, ``verify_proto_skills``,
               ``promote_proto_skills``)
  Materialize — Promote ``__NEW__`` clusters to real skills
              (``SkillBankAgent.materialize_new_skills``)
  Phase 5   — Distill execution hints
              (``SkillBankAgent.distill_execution_hints``)
  Protocols — Update protocols from evidence
              (``SkillBankAgent.update_protocols``)
  Evaluation — Skill quality assessment (coherence, granularity, …)
              (``SkillBankAgent.run_evaluation``)

Then generates:
  - Qwen3-8B skill names, descriptions, and RAG summaries
  - Per-game ``skill_bank.jsonl`` and ``skill_catalog.json``
  - Cross-game archetype aggregation (``skill_archetypes.json``)
  - Combined ``skill_catalog_all.json``
  - Per-stage I/O recordings (``stage_io_log.json``)
  - Cold-start I/O records (``coldstart_io_all.jsonl``)
  - Teacher I/O records (``teacher_io_coldstart.jsonl``)

Usage (from Game-AI-Agent root):

    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
    export VLLM_BASE_URL="http://localhost:8000/v1"

    # All games
    python -m scripts.qwen3_skillbank_agent

    # Specific games
    python -m scripts.qwen3_skillbank_agent --games tetris twenty_forty_eight

    # Quick test: one episode per game
    python -m scripts.qwen3_skillbank_agent --one_per_game -v

    # Custom input/output dirs
    python -m scripts.qwen3_skillbank_agent \\
        --input_dir labeling/output/gpt54 \\
        --output_dir test_rollout/skillbank_agent_grpo

    # Enable GRPO training with LoRA
    python -m scripts.qwen3_skillbank_agent \\
        --use_grpo --local_model Qwen/Qwen3-8B --one_per_game -v

    # Re-segment against seeded bank
    python -m scripts.qwen3_skillbank_agent --resegment --one_per_game

    # Resume interrupted run
    python -m scripts.qwen3_skillbank_agent --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from decision_agents.agent_helper import (
    strip_think_tags,
    HARD_SUMMARY_CHAR_LIMIT,
    SUBGOAL_TAGS,
)

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

# ── skill_agents_grpo imports ────────────────────────────────────────
from skill_agents_grpo.pipeline import SkillBankAgent, PipelineConfig
from skill_agents_grpo.stage3_mvp.schemas import (
    ExecutionHint,
    SegmentRecord,
    SkillEffectsContract,
    SubEpisodeRef,
)
from data_structure.experience import Episode, Experience, SubTask_Experience

try:
    from skill_agents_grpo.infer_segmentation.episode_adapter import (
        _extract_predicates,
    )
except ImportError:
    _extract_predicates = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-8B"

DEFAULT_INPUT_DIR = CODEBASE_ROOT / "labeling" / "output" / "gpt54"
DEFAULT_OUTPUT_DIR = CODEBASE_ROOT / "test_rollout" / "skillbank_agent_grpo"

_SUBGOAL_TAG_SET = frozenset(SUBGOAL_TAGS)


# ---------------------------------------------------------------------------
# GRPO / local-model helpers
# ---------------------------------------------------------------------------

def setup_local_model(
    model_name_or_path: str,
    adapter_dir: Optional[str] = None,
) -> "MultiLoraSkillBankLLM":
    """Instantiate the local Qwen3-8B model with optional LoRA adapters."""
    from skill_agents_grpo.lora.config import MultiLoraConfig
    from skill_agents_grpo.lora.model import MultiLoraSkillBankLLM
    from skill_agents_grpo.lora.skill_function import SkillFunction

    adapter_paths: Dict[str, str] = {}
    if adapter_dir:
        _ad = Path(adapter_dir)
        for fn in (SkillFunction.SEGMENT, SkillFunction.CONTRACT, SkillFunction.CURATOR):
            candidate = _ad / fn.value
            if candidate.exists():
                adapter_paths[fn.value] = str(candidate)

    cfg = MultiLoraConfig(
        base_model_name_or_path=model_name_or_path,
        adapter_paths=adapter_paths,
        allow_fallback_to_base_model=True,
    )
    llm = MultiLoraSkillBankLLM(cfg)
    MultiLoraSkillBankLLM.set_shared_instance(llm)
    print(f"  Local model: {model_name_or_path}")
    if adapter_paths:
        print(f"  LoRA adapters: {list(adapter_paths.keys())}")
    return llm


def setup_grpo_orchestrator(
    llm: "MultiLoraSkillBankLLM",
    group_size: int = 4,
) -> "GRPOOrchestrator":
    """Create and return a GRPO orchestrator."""
    from skill_agents_grpo.grpo.orchestrator import GRPOOrchestrator
    from skill_agents_grpo.grpo.config import GRPOConfig, StageGRPOConfig
    from skill_agents_grpo.lora.skill_function import SkillFunction

    grpo_cfg = GRPOConfig(stage_configs={
        SkillFunction.SEGMENT.value: StageGRPOConfig(
            group_size=group_size, kl_coeff=0.02, lr=3e-5,
            epochs_per_batch=3, temperature=0.7,
        ),
        SkillFunction.CONTRACT.value: StageGRPOConfig(
            group_size=group_size, kl_coeff=0.05, lr=5e-5,
            epochs_per_batch=2, temperature=0.7,
        ),
        SkillFunction.CURATOR.value: StageGRPOConfig(
            group_size=group_size, kl_coeff=0.05, lr=5e-5,
            epochs_per_batch=2, temperature=0.7,
        ),
    })
    orch = GRPOOrchestrator(llm, grpo_cfg)
    print(f"  GRPO orchestrator created (G={group_size})")
    return orch


def save_lora_adapters(
    llm: "MultiLoraSkillBankLLM",
    adapter_dir: str,
) -> None:
    """Save all loaded LoRA adapters to disk."""
    from skill_agents_grpo.lora.skill_function import SkillFunction

    _ad = Path(adapter_dir)
    _ad.mkdir(parents=True, exist_ok=True)

    if not llm.is_loaded or not llm._is_peft_model:
        return

    for fn in (SkillFunction.SEGMENT, SkillFunction.CONTRACT, SkillFunction.CURATOR):
        name = fn.adapter_name
        if name not in llm._loaded_adapters:
            continue
        save_path = _ad / fn.value
        save_path.mkdir(parents=True, exist_ok=True)
        try:
            llm._model.set_adapter(name)
            llm._model.save_pretrained(str(save_path))
            print(f"    Saved adapter '{name}' → {save_path}")
        except Exception as exc:
            print(f"    [WARN] Failed to save adapter '{name}': {exc}")


_TAG_ALIASES: Dict[str, str] = {
    "PLACE": "SETUP", "DROP": "EXECUTE", "MOVE": "NAVIGATE",
    "SWAP": "EXECUTE", "PUSH": "NAVIGATE", "JUMP": "NAVIGATE",
    "MATCH": "CLEAR", "PLAN": "SETUP", "ARRANGE": "SETUP",
    "ROTATE": "SETUP", "ORGANIZE": "OPTIMIZE", "SCORE": "EXECUTE",
    "PROTECT": "DEFEND", "GRAB": "COLLECT", "FLEE": "SURVIVE",
    "RUN": "NAVIGATE", "CREATE": "BUILD", "FIND": "EXPLORE",
    "FIX": "OPTIMIZE", "ALIGN": "POSITION", "TARGET": "ATTACK",
    "SECURE": "DEFEND", "EXPAND": "ATTACK", "RETREAT": "DEFEND",
}

_TAG_RE = re.compile(r"\[(\w+)\]\s*")


# ═══════════════════════════════════════════════════════════════════════
# Stage I/O recording
# ═══════════════════════════════════════════════════════════════════════

class StageIORecord:
    """Captures serialisable I/O metadata for one pipeline stage invocation."""

    def __init__(self, stage_name: str,
                 on_finish: Optional[Callable[["StageIORecord"], None]] = None) -> None:
        self.stage_name = stage_name
        self.timestamp_start: Optional[str] = None
        self.timestamp_end: Optional[str] = None
        self.elapsed_seconds: float = 0.0
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self._t0: float = 0.0
        self._on_finish = on_finish

    def start(self) -> "StageIORecord":
        self._t0 = time.time()
        self.timestamp_start = datetime.now().isoformat()
        return self

    def finish(self) -> "StageIORecord":
        self.elapsed_seconds = round(time.time() - self._t0, 3)
        self.timestamp_end = datetime.now().isoformat()
        if self._on_finish is not None:
            try:
                self._on_finish(self)
            except Exception:
                pass
        return self

    def record_input(self, key: str, value: Any) -> None:
        self.inputs[key] = _safe_serialize(value)

    def record_output(self, key: str, value: Any) -> None:
        self.outputs[key] = _safe_serialize(value)

    def record_error(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "stage": self.stage_name,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "elapsed_seconds": self.elapsed_seconds,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        if self.error:
            d["error"] = self.error
        return d


def _safe_serialize(obj: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """Convert arbitrary objects to JSON-safe form, truncating depth."""
    if depth > max_depth:
        return f"<depth-limit: {type(obj).__name__}>"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:500] if len(obj) > 500 else obj
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)[:50]
    if isinstance(obj, (list, tuple)):
        if len(obj) > 20:
            return {
                "_type": type(obj).__name__,
                "_len": len(obj),
                "_sample": [_safe_serialize(x, depth + 1) for x in obj[:5]],
            }
        return [_safe_serialize(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        if len(obj) > 30:
            return {
                "_type": "dict",
                "_len": len(obj),
                "_keys_sample": list(obj.keys())[:10],
            }
        return {str(k): _safe_serialize(v, depth + 1) for k, v in list(obj.items())[:30]}
    if hasattr(obj, "to_dict"):
        try:
            return _safe_serialize(obj.to_dict(), depth + 1)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {
            "_type": type(obj).__name__,
            "_fields": {
                k: _safe_serialize(v, depth + 1)
                for k, v in list(vars(obj).items())[:15]
                if not k.startswith("_")
            },
        }
    return f"<{type(obj).__name__}>"


class StageIOLog:
    """Accumulates stage I/O records for a single game run."""

    def __init__(self, game_name: str, model: str,
                 on_finish: Optional[Callable[[StageIORecord], None]] = None) -> None:
        self.game_name = game_name
        self.model = model
        self.records: List[StageIORecord] = []
        self._on_finish = on_finish

    def new_record(self, stage_name: str) -> StageIORecord:
        rec = StageIORecord(stage_name, on_finish=self._on_finish)
        self.records.append(rec)
        return rec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game": self.game_name,
            "model": self.model,
            "pipeline": "skill_agents_grpo",
            "timestamp": datetime.now().isoformat(),
            "n_stages": len(self.records),
            "stages": [r.to_dict() for r in self.records],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint / resume support
# ═══════════════════════════════════════════════════════════════════════

_CHECKPOINT_FILENAME = "extraction_checkpoint.json"


class ExtractionCheckpoint:
    """Track which games / episodes have completed so the run can resume."""

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / _CHECKPOINT_FILENAME
        self._data: Dict[str, Any] = {
            "completed_games": [],
            "in_progress_game": None,
            "completed_episodes": {},
            "stage_reached": {},
        }
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data.update(json.load(f))
            except Exception:
                pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        tmp.replace(self._path)

    def is_game_complete(self, game: str) -> bool:
        return game in self._data["completed_games"]

    def completed_episode_count(self, game: str) -> int:
        return self._data["completed_episodes"].get(game, 0)

    def stage_reached(self, game: str) -> Optional[str]:
        return self._data["stage_reached"].get(game)

    def mark_episode_done(self, game: str, episode_idx: int, stage: str = "stage_1_2") -> None:
        self._data["in_progress_game"] = game
        self._data["completed_episodes"][game] = episode_idx + 1
        self._data["stage_reached"][game] = stage
        self._save()

    def mark_stage(self, game: str, stage: str) -> None:
        self._data["stage_reached"][game] = stage
        self._save()

    def mark_game_complete(self, game: str) -> None:
        if game not in self._data["completed_games"]:
            self._data["completed_games"].append(game)
        self._data["in_progress_game"] = None
        self._data["stage_reached"].pop(game, None)
        self._data["completed_episodes"].pop(game, None)
        self._save()

    def reset(self) -> None:
        self._data = {
            "completed_games": [],
            "in_progress_game": None,
            "completed_episodes": {},
            "stage_reached": {},
        }
        if self._path.exists():
            self._path.unlink()

    @property
    def path(self) -> Path:
        return self._path


# ═══════════════════════════════════════════════════════════════════════
# LLM call I/O recording
# ═══════════════════════════════════════════════════════════════════════

_llm_call_log: List[Dict[str, Any]] = []


def _flush_llm_calls() -> List[Dict[str, Any]]:
    """Return and clear accumulated LLM call records since last flush."""
    global _llm_call_log
    calls = list(_llm_call_log)
    _llm_call_log.clear()
    return calls


def _reset_llm_call_log() -> None:
    """Clear all accumulated LLM call records."""
    _llm_call_log.clear()


# ═══════════════════════════════════════════════════════════════════════
# LLM helpers — Qwen3-8B via vLLM (ask_model) with /no_think
# ═══════════════════════════════════════════════════════════════════════

def _ask_llm(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 400,
    _caller: str = "",
) -> Optional[str]:
    """Call Qwen3-8B via ask_model and strip think tags.

    Appends /no_think to disable Qwen3's thinking mode so the full
    token budget goes to the actual structured output.
    """
    call_t0 = time.time()
    full_prompt = prompt.rstrip() + "\n/no_think"
    result = None
    error_msg = None

    if ask_model is not None:
        try:
            raw = ask_model(
                full_prompt, model=model,
                temperature=temperature, max_tokens=max_tokens,
            )
            if raw and not raw.startswith("Error"):
                result = strip_think_tags(raw).strip()
            elif raw:
                error_msg = raw[:500]
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

    call_record: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "caller": _caller,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_chars": len(prompt),
        "prompt": prompt[:3000],
        "response_chars": len(result) if result else 0,
        "response": result[:3000] if result else None,
        "elapsed_s": round(time.time() - call_t0, 3),
    }
    if error_msg:
        call_record["error"] = error_msg
    _llm_call_log.append(call_record)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Episode loading / conversion
# ═══════════════════════════════════════════════════════════════════════

def find_episode_files(
    input_dir: Path,
    games: Optional[List[str]] = None,
) -> Dict[str, List[Path]]:
    """Discover episode JSON files grouped by game name."""
    game_files: Dict[str, List[Path]] = {}
    if not input_dir.exists():
        return game_files
    for gd in sorted(input_dir.iterdir()):
        if not gd.is_dir():
            continue
        game_name = gd.name
        if games and game_name not in games:
            continue
        eps = sorted(
            fp for fp in gd.glob("episode_*.json")
            if fp.name != "episode_buffer.json"
        )
        if eps:
            game_files[game_name] = eps
    return game_files


def load_episode(filepath: Path) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def dict_to_episode(episode_data: Dict[str, Any]) -> Episode:
    """Convert a labeled episode dict to an Episode object for skill_agents_grpo."""
    experiences = []
    for exp_d in episode_data.get("experiences", []):
        exp = Experience(
            state=exp_d.get("state", ""),
            action=exp_d.get("action", ""),
            reward=exp_d.get("reward", 0.0),
            next_state=exp_d.get("next_state", ""),
            done=exp_d.get("done", False),
            intentions=exp_d.get("intentions"),
            tasks=exp_d.get("tasks"),
            sub_tasks=exp_d.get("sub_tasks"),
        )
        exp.idx = exp_d.get("idx")
        exp.summary = exp_d.get("summary")
        exp.summary_state = exp_d.get("summary_state")
        exp.raw_state = exp_d.get("raw_state")
        exp.raw_next_state = exp_d.get("raw_next_state")
        exp.available_actions = exp_d.get("available_actions")
        exp.interface = exp_d.get("interface")
        exp.action_type = exp_d.get("action_type")
        experiences.append(exp)

    task = episode_data.get("task", episode_data.get("game_name", ""))
    return Episode(
        experiences=experiences,
        task=task,
        episode_id=episode_data.get("episode_id"),
        env_name=episode_data.get("env_name", "gamingagent"),
        game_name=episode_data.get("game_name", ""),
    )


# ═══════════════════════════════════════════════════════════════════════
# Skill naming / description via Qwen3-8B
# ═══════════════════════════════════════════════════════════════════════

def generate_skill_name(
    skill_id: str,
    contract: SkillEffectsContract,
    game_name: str,
    sample_intentions: List[str],
    model: str = DEFAULT_MODEL,
) -> Tuple[str, str]:
    """Ask Qwen3-8B to generate a short skill name and RAG summary."""
    eff_add_str = ", ".join(sorted(contract.eff_add)[:8]) if contract.eff_add else "none"
    eff_del_str = ", ".join(sorted(contract.eff_del)[:8]) if contract.eff_del else "none"
    eff_event_str = ", ".join(sorted(contract.eff_event)[:5]) if contract.eff_event else "none"
    intentions_str = " | ".join(sample_intentions[:5]) if sample_intentions else "n/a"

    prompt = (
        f"You are a game-AI skill naming expert. Read the skill data below and "
        f"generate a concrete, game-specific name and RAG summary.\n\n"
        f"Game: {game_name}\n"
        f"Skill ID: {skill_id}\n"
        f"Effects added: {eff_add_str}\n"
        f"Effects removed: {eff_del_str}\n"
        f"Events: {eff_event_str}\n"
        f"Sample intentions from segments: {intentions_str}\n\n"
        f"RULES:\n"
        f"- The NAME must be a concrete imperative verb phrase (2-5 words) "
        f"describing the actual game action, NOT the skill_id.\n"
        f"- The SUMMARY must include context= describing WHEN and WHY to use this skill.\n"
        f"- Be specific to {game_name}. Reference game objects, mechanics, or goals.\n\n"
        f"Reply in this EXACT format (two lines only):\n"
        f"NAME: <2-5 word imperative verb phrase>\n"
        f"SUMMARY: game={game_name} | skill=<name> | effects=<top effects> | "
        f"context=<when to use this skill and why, be specific>\n"
    )

    call_t0 = time.time()
    result = _ask_llm(prompt, model=model, max_tokens=200, temperature=0.3,
                      _caller=f"generate_skill_name:{skill_id}")

    name = skill_id.replace("_", " ").title()
    rag_summary = f"game={game_name} | skill={skill_id} | eff_add={eff_add_str}"

    if result:
        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("NAME:"):
                parsed_name = line[5:].strip().strip('"').strip("'")
                if 2 <= len(parsed_name) <= 60:
                    name = parsed_name
            elif line.upper().startswith("SUMMARY:"):
                parsed_summary = line[8:].strip().strip('"').strip("'")
                if len(parsed_summary) > 10:
                    rag_summary = parsed_summary[:HARD_SUMMARY_CHAR_LIMIT]

    from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord
    record_io(ColdStartRecord(
        module="skill_naming",
        function="generate_skill_name",
        prompt=prompt,
        response=result or "",
        parsed={"name": name, "rag_summary": rag_summary},
        model=model,
        temperature=0.3,
        max_tokens=200,
        elapsed_s=round(time.time() - call_t0, 3),
        skill_id=skill_id,
        extra={"game": game_name},
    ))

    return name, rag_summary


def generate_skill_protocol(
    skill_id: str,
    name: str,
    description: str,
    contract: SkillEffectsContract,
    game_name: str,
    sample_intentions: List[str],
    sample_states: List[str],
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Ask Qwen3-8B to generate actionable protocol fields for a skill."""
    eff_add_str = ", ".join(sorted(contract.eff_add)[:6]) if contract.eff_add else "none"
    eff_del_str = ", ".join(sorted(contract.eff_del)[:6]) if contract.eff_del else "none"
    eff_event_str = ", ".join(sorted(contract.eff_event)[:5]) if contract.eff_event else "none"
    intentions_str = " | ".join(sample_intentions[:4]) if sample_intentions else "n/a"
    states_str = " // ".join(s[:100] for s in sample_states[:3]) if sample_states else "n/a"

    prompt = (
        f"You are a game-AI protocol designer. Generate a concrete, game-specific "
        f"execution protocol for the skill below.\n\n"
        f"Game: {game_name}\n"
        f"Skill: {name} ({skill_id})\n"
        f"Description: {description}\n"
        f"Effects added: {eff_add_str}\n"
        f"Effects removed: {eff_del_str}\n"
        f"Events: {eff_event_str}\n"
        f"Sample intentions: {intentions_str}\n"
        f"Sample states: {states_str}\n\n"
        f"RULES:\n"
        f"- PRECONDITIONS: describe the specific game situation when this skill "
        f"should be invoked (reference game phase, objects, threats).\n"
        f"- STEPS: list 3-7 concrete actions the agent should take, specific to "
        f"{game_name}. Do NOT write generic steps like 'Achieve: X'.\n"
        f"- SUCCESS_CRITERIA: observable conditions proving the skill worked.\n"
        f"- ABORT_CRITERIA: when to give up (specific to this skill and {game_name}).\n\n"
        f"Reply in this EXACT format (one item per line within each section):\n"
        f"PRECONDITIONS:\n- <when this skill should be invoked>\n"
        f"STEPS:\n- <step 1>\n- <step 2>\n- ...\n"
        f"SUCCESS_CRITERIA:\n- <how to know the skill succeeded>\n"
        f"ABORT_CRITERIA:\n- <game-specific condition when to abandon this skill>\n"
    )

    result = _ask_llm(prompt, model=model, max_tokens=800, temperature=0.3,
                      _caller=f"generate_skill_protocol:{skill_id}")

    preconditions: List[str] = []
    steps: List[str] = []
    success_criteria: List[str] = []
    abort_criteria: List[str] = []

    if result:
        current_section = None
        for line in result.split("\n"):
            line = line.strip()
            upper = line.upper().rstrip(":")
            if upper in ("PRECONDITIONS", "PRECONDITION"):
                current_section = "preconditions"
                continue
            elif upper in ("STEPS", "STEP"):
                current_section = "steps"
                continue
            elif upper in ("SUCCESS_CRITERIA", "SUCCESS CRITERIA", "SUCCESS"):
                current_section = "success_criteria"
                continue
            elif upper in ("ABORT_CRITERIA", "ABORT CRITERIA", "ABORT"):
                current_section = "abort_criteria"
                continue

            if line.startswith("- "):
                line = line[2:].strip()
            elif line.startswith("* "):
                line = line[2:].strip()
            elif line and line[0].isdigit() and ". " in line[:4]:
                line = line.split(". ", 1)[1].strip()

            if not line:
                continue

            if current_section == "preconditions" and len(preconditions) < 3:
                preconditions.append(line)
            elif current_section == "steps" and len(steps) < 7:
                steps.append(line)
            elif current_section == "success_criteria" and len(success_criteria) < 3:
                success_criteria.append(line)
            elif current_section == "abort_criteria" and len(abort_criteria) < 3:
                abort_criteria.append(line)

    def _trim_incomplete(items: List[str]) -> List[str]:
        if not items:
            return items
        last = items[-1].rstrip()
        if last and last[-1] not in ".!?)]\u2019\u201d":
            cut = last.rfind(".")
            if cut > 20:
                items[-1] = last[:cut + 1]
            elif len(items) > 1:
                items = items[:-1]
        return items

    steps = _trim_incomplete(steps)
    preconditions = _trim_incomplete(preconditions)
    success_criteria = _trim_incomplete(success_criteria)
    abort_criteria = _trim_incomplete(abort_criteria)

    _GENERIC_ABORT_PHRASES = {
        "no progress after expected duration",
        "no progress after duration",
        "no progress",
    }
    abort_criteria = [
        a for a in abort_criteria
        if a.strip().rstrip(".").lower() not in _GENERIC_ABORT_PHRASES
    ]

    if not steps:
        if contract.eff_add:
            for lit in sorted(contract.eff_add)[:4]:
                steps.append(f"Achieve: {lit}")
        if contract.eff_event:
            for lit in sorted(contract.eff_event)[:2]:
                steps.append(f"Trigger: {lit}")
        if not steps:
            steps = [f"Execute {name} actions as needed"]

    if not preconditions and sample_intentions:
        tag_match = _TAG_RE.match(sample_intentions[0].strip())
        if tag_match:
            preconditions = [f"Situation calls for [{tag_match.group(1).upper()}] action"]

    if not success_criteria:
        if contract.eff_add:
            success_criteria = [f"{lit} is achieved" for lit in sorted(contract.eff_add)[:2]]
        elif contract.eff_event:
            success_criteria = [f"{ev} observed" for ev in sorted(contract.eff_event)[:2]]
        else:
            success_criteria = [f"{name} completed successfully"]

    if not abort_criteria:
        abort_criteria = [f"The game state in {game_name} no longer permits {name} actions"]

    return {
        "preconditions": preconditions,
        "steps": steps,
        "success_criteria": success_criteria,
        "abort_criteria": abort_criteria,
    }


def generate_skill_description(
    skill_id: str,
    name: str,
    contract: SkillEffectsContract,
    game_name: str,
    sample_states: List[str],
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask Qwen3-8B to generate a 1-2 sentence description for the skill."""
    eff_str = ", ".join(sorted(contract.eff_add | contract.eff_del)[:10])
    states_str = " // ".join(s[:100] for s in sample_states[:3]) if sample_states else "n/a"

    prompt = (
        f"You are a game-AI skill documentation expert.\n\n"
        f"Game: {game_name}\nSkill: {name} ({skill_id})\n"
        f"Effects: {eff_str}\n"
        f"Sample states where skill was executed: {states_str}\n\n"
        f"Write exactly 1-2 sentences describing what this skill does and when to "
        f"use it. Be concrete and specific to {game_name} — reference actual game "
        f"objects, mechanics, positions, or goals. Max 40 words.\n"
        f"Do NOT repeat the skill_id or say 'applies X'. Describe the actual action.\n"
        f"Description:"
    )

    call_t0 = time.time()
    result = _ask_llm(prompt, model=model, max_tokens=200, temperature=0.3,
                      _caller=f"generate_skill_description:{skill_id}")
    desc = f"Skill '{name}' in {game_name}: applies {eff_str[:80]}."
    if result:
        raw_desc = result.split("\n")[0].strip().strip('"').strip("'")
        if len(raw_desc) > 250:
            cut = raw_desc[:250].rfind(".")
            raw_desc = raw_desc[:cut + 1] if cut > 80 else raw_desc[:250]
        if raw_desc and raw_desc[-1] not in ".!?":
            cut = raw_desc.rfind(".")
            if cut > 40:
                raw_desc = raw_desc[:cut + 1]
        desc = raw_desc

    from skill_agents_grpo.coldstart_io import record_io, ColdStartRecord
    record_io(ColdStartRecord(
        module="skill_naming",
        function="generate_skill_description",
        prompt=prompt,
        response=result or "",
        parsed={"description": desc},
        model=model,
        temperature=0.3,
        max_tokens=200,
        elapsed_s=round(time.time() - call_t0, 3),
        skill_id=skill_id,
        extra={"game": game_name},
    ))

    return desc


# ═══════════════════════════════════════════════════════════════════════
# Predicate effects computation
# ═══════════════════════════════════════════════════════════════════════

def _compute_predicate_effects(
    predicates: List[Optional[dict]],
    start: int,
    end: int,
    p_thresh: float = 0.5,
) -> Tuple[set, set, set]:
    eff_add: set = set()
    eff_del: set = set()
    eff_event: set = set()
    if not predicates:
        return eff_add, eff_del, eff_event

    p_start = predicates[start] if start < len(predicates) else None
    p_end = predicates[min(end - 1, len(predicates) - 1)] if end > 0 else None
    if p_start is None or p_end is None:
        return eff_add, eff_del, eff_event

    for k, v in p_end.items():
        if isinstance(v, (int, float)) and v >= p_thresh:
            sv = p_start.get(k)
            if sv is None or (isinstance(sv, (int, float)) and sv < p_thresh):
                eff_add.add(k)

    for k, v in p_start.items():
        if isinstance(v, (int, float)) and v >= p_thresh:
            ev = p_end.get(k)
            if ev is not None and isinstance(ev, (int, float)) and ev < p_thresh:
                eff_del.add(k)

    for t in range(start, min(end, len(predicates))):
        p = predicates[t]
        if p:
            for k, v in p.items():
                if k.startswith("tag_") and isinstance(v, (int, float)) and v >= p_thresh:
                    eff_event.add(k)

    return eff_add, eff_del, eff_event


# ═══════════════════════════════════════════════════════════════════════
# Intention-based fallback segmentation
# ═══════════════════════════════════════════════════════════════════════

def _compute_sub_episode_quality(
    segment_exps: list,
    outcome_exps: Optional[list],
) -> float:
    """Compute a [0, 1] quality score for a fallback sub-episode."""
    if not segment_exps:
        return 0.0
    seg_reward = sum(getattr(e, "reward", 0.0) or 0.0 for e in segment_exps)
    n_steps = len(segment_exps)
    reward_score = min(1.0, max(0.0, seg_reward / max(n_steps, 1)))

    outcome_score = 0.0
    if outcome_exps:
        oc_reward = sum(getattr(e, "reward", 0.0) or 0.0 for e in outcome_exps)
        outcome_score = 0.5 if oc_reward > 0 else 0.0

    length_score = min(1.0, n_steps / 10.0) if n_steps >= 2 else 0.2

    return round(min(1.0, 0.4 * reward_score + 0.3 * outcome_score + 0.3 * length_score), 3)


def _build_sub_episodes_from_tags(
    tag_segments: List[Dict[str, Any]],
    episodes: List[Episode],
    game_name: str,
    outcome_length: int = 5,
) -> List[SubTask_Experience]:
    sub_episodes: List[SubTask_Experience] = []
    for seg in tag_segments:
        ep_idx = seg["ep_idx"]
        if ep_idx >= len(episodes):
            continue
        ep = episodes[ep_idx]
        exps = ep.experiences
        start, end = seg["start"], seg["end"]
        segment_exps = exps[start:end]
        if not segment_exps:
            continue
        outcome_start = end
        outcome_end = min(end + outcome_length, len(exps))
        outcome_exps = (
            exps[outcome_start:outcome_end]
            if outcome_start < outcome_end
            else None
        )
        seg_id = f"tag_{game_name}_{seg.get('tag', 'UNK')}_{ep_idx}_{start}_{end}"
        sub_ep = SubTask_Experience(
            sub_task=seg.get("skill_id", seg.get("tag", "EXECUTE")),
            final_goal=ep.task,
            experiences=segment_exps,
            outcome=outcome_exps,
            seg_id=seg_id,
        )
        sub_ep.quality_score = _compute_sub_episode_quality(segment_exps, outcome_exps)
        sub_episodes.append(sub_ep)
    return sub_episodes


def _link_sub_episodes_to_skills(
    agent: SkillBankAgent,
    all_sub_episodes: List[SubTask_Experience],
    verbose: bool = False,
) -> int:
    """Create SubEpisodeRef pointers on each Skill from SubTask_Experience objects."""
    linked = 0
    skill_refs: Dict[str, List[SubEpisodeRef]] = defaultdict(list)

    active_ids = [
        sid for sid in agent.skill_ids
        if not (agent.bank.get_skill(sid) or type("", (), {"retired": True})()).retired
    ]
    new_remap_target = active_ids[0] if len(active_ids) == 1 else None

    for se in all_sub_episodes:
        skill_id = se.sub_task
        if not skill_id:
            continue
        if skill_id == "__NEW__" and new_remap_target:
            skill_id = new_remap_target

        cum_reward = sum(
            getattr(e, "reward", 0.0) or 0.0
            for e in (se.sub_task_experience or [])
        )
        n_steps = len(se.sub_task_experience) if se.sub_task_experience else 0
        outcome_exps = se.outcome_experiences if hasattr(se, "outcome_experiences") else None
        oc_class = getattr(se, "outcome_classification", None)
        outcome = "partial"
        if oc_class == "success":
            outcome = "success"
        elif oc_class == "failure":
            outcome = "failure"
        elif outcome_exps:
            outcome_reward = sum(getattr(e, "reward", 0.0) or 0.0 for e in outcome_exps)
            if outcome_reward > 0:
                outcome = "success"
        if outcome == "partial" and cum_reward > 0:
            outcome = "success"

        intent_tags = []
        for e in (se.sub_task_experience or []):
            intent = getattr(e, "intentions", "") or ""
            m = _TAG_RE.match(intent.strip())
            if m:
                intent_tags.append(m.group(1).upper())

        summary_parts = []
        if se.sub_task_experience:
            first_s = getattr(se.sub_task_experience[0], "summary", "") or ""
            if first_s:
                summary_parts.append(first_s[:80])
        if not summary_parts:
            summary_parts.append(f"{skill_id}: {n_steps} steps")

        ref = SubEpisodeRef(
            episode_id=se.episode_id if hasattr(se, "episode_id") else "",
            seg_start=se.sub_task_experience[0].idx if se.sub_task_experience and hasattr(se.sub_task_experience[0], "idx") and se.sub_task_experience[0].idx is not None else 0,
            seg_end=(se.sub_task_experience[-1].idx or 0) + 1 if se.sub_task_experience and hasattr(se.sub_task_experience[-1], "idx") and se.sub_task_experience[-1].idx is not None else n_steps,
            rollout_source=se.seg_id if hasattr(se, "seg_id") else "",
            summary=summary_parts[0],
            intention_tags=intent_tags[:10],
            outcome=outcome,
            cumulative_reward=cum_reward,
            quality_score=se.quality_score if hasattr(se, "quality_score") and se.quality_score else 0.0,
        )
        skill_refs[skill_id].append(ref)

    for sid, refs in skill_refs.items():
        skill = agent.bank.get_skill(sid)
        if skill is None:
            continue
        existing_scores: Dict[tuple, float] = {}
        for se in (skill.sub_episodes or []):
            key = (getattr(se, "episode_id", ""), getattr(se, "seg_start", -1))
            if se.quality_score > 0:
                existing_scores[key] = se.quality_score
        if existing_scores:
            for ref in refs:
                key = (getattr(ref, "episode_id", ""), getattr(ref, "seg_start", -1))
                if key in existing_scores:
                    ref.quality_score = existing_scores[key]
        skill.sub_episodes = refs
        skill.n_instances = max(skill.n_instances, len(refs))
        agent.bank.add_or_update_skill(skill)
        linked += len(refs)

    if verbose:
        print(f"    Linked {linked} sub-episode ref(s) across {len(skill_refs)} skill(s)")
    return linked


def intention_based_segmentation(
    episodes_data: List[Dict[str, Any]],
    game_name: str,
    episodes: Optional[List[Episode]] = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    outcome_length: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[SubTask_Experience]]:
    """Fallback segmentation using intention tag transitions."""
    tag_segments: List[Dict[str, Any]] = []

    for ep_idx, ep_data in enumerate(episodes_data):
        exps = ep_data.get("experiences", [])
        if not exps:
            continue

        current_tag = None
        seg_start = 0

        for i, exp in enumerate(exps):
            intent = exp.get("intentions", "")
            m = _TAG_RE.match(intent.strip())
            tag = m.group(1).upper() if m else "EXECUTE"
            if tag not in _SUBGOAL_TAG_SET:
                tag = _TAG_ALIASES.get(tag, "EXECUTE")

            if tag != current_tag:
                if current_tag is not None and i > seg_start:
                    tag_segments.append({
                        "ep_idx": ep_idx,
                        "tag": current_tag,
                        "start": seg_start,
                        "end": i,
                        "intentions": [
                            exps[t].get("intentions", "")
                            for t in range(seg_start, min(i, len(exps)))
                        ],
                        "states": [
                            str(exps[t].get("summary_state", ""))[:150]
                            for t in range(seg_start, min(i, len(exps)))
                        ],
                    })
                current_tag = tag
                seg_start = i

        if current_tag is not None and len(exps) > seg_start:
            tag_segments.append({
                "ep_idx": ep_idx,
                "tag": current_tag,
                "start": seg_start,
                "end": len(exps),
                "intentions": [
                    exps[t].get("intentions", "")
                    for t in range(seg_start, len(exps))
                ],
                "states": [
                    str(exps[t].get("summary_state", ""))[:150]
                    for t in range(seg_start, len(exps))
                ],
            })

    if not tag_segments:
        return [], {}, []

    episode_predicates: Dict[int, List[Optional[dict]]] = {}
    if episodes and _extract_predicates is not None:
        for ep_idx, ep in enumerate(episodes):
            try:
                episode_predicates[ep_idx] = _extract_predicates(ep.experiences)
            except Exception:
                pass

    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seg in tag_segments:
        by_tag[seg["tag"]].append(seg)

    if verbose:
        print(f"    Intention-based segmentation: {len(tag_segments)} segments, {len(by_tag)} unique tags")

    skill_catalog: Dict[str, Dict[str, Any]] = {}
    skill_idx = 0

    for tag, segs in sorted(by_tag.items()):
        skill_id = f"skill_{game_name}_{tag.lower()}_{skill_idx}"
        skill_idx += 1

        total_steps = sum(s["end"] - s["start"] for s in segs)
        sample_intentions: List[str] = []
        sample_states: List[str] = []
        for s in segs[:5]:
            sample_intentions.extend(s["intentions"][:3])
            sample_states.extend(s["states"][:3])

        agg_add: Dict[str, int] = defaultdict(int)
        agg_del: Dict[str, int] = defaultdict(int)
        agg_event: set = set()
        n_pred_segs = 0
        for s in segs:
            ep_preds = episode_predicates.get(s["ep_idx"])
            if not ep_preds:
                continue
            ea, ed, ee = _compute_predicate_effects(ep_preds, s["start"], s["end"])
            if ea or ed or ee:
                n_pred_segs += 1
                for k in ea:
                    agg_add[k] += 1
                for k in ed:
                    agg_del[k] += 1
                agg_event |= ee

        min_freq = 0.3
        real_eff_add: set = set()
        real_eff_del: set = set()
        if n_pred_segs > 0:
            for k, cnt in agg_add.items():
                if cnt / n_pred_segs >= min_freq:
                    real_eff_add.add(k)
            for k, cnt in agg_del.items():
                if cnt / n_pred_segs >= min_freq:
                    real_eff_del.add(k)

        contract = SkillEffectsContract(
            skill_id=skill_id,
            eff_add=real_eff_add if real_eff_add else {f"{tag.lower()}_completed"},
            eff_del=real_eff_del,
            eff_event=agg_event if agg_event else {f"tag_{tag.lower()}"},
            n_instances=len(segs),
        )

        name, rag_summary = generate_skill_name(
            skill_id, contract, game_name, sample_intentions, model=model,
        )
        description = generate_skill_description(
            skill_id, name, contract, game_name, sample_states, model=model,
        )

        contract.name = name
        contract.description = description

        skill_catalog[skill_id] = {
            "skill_id": skill_id,
            "name": name,
            "summary": rag_summary,
            "description": description,
            "tag": tag,
            "eff_add": sorted(contract.eff_add),
            "eff_del": sorted(contract.eff_del),
            "eff_event": sorted(contract.eff_event),
            "n_instances": len(segs),
            "total_steps": total_steps,
            "version": 1,
        }

        if verbose:
            print(f"      [{tag}] {name} — {len(segs)} segment(s), {total_steps} steps")

        for s in segs:
            s["skill_id"] = skill_id
            s["skill_name"] = name
            s["skill_summary"] = rag_summary
            s["description"] = description

    # Deduplicate near-identical skills
    names_seen: Dict[str, str] = {}
    merge_map: Dict[str, str] = {}
    for sid, entry in list(skill_catalog.items()):
        norm_name = entry["name"].strip().lower()
        if norm_name in names_seen:
            canonical = names_seen[norm_name]
            merge_map[sid] = canonical
            skill_catalog[canonical]["n_instances"] += entry["n_instances"]
            skill_catalog[canonical]["total_steps"] += entry["total_steps"]
            del skill_catalog[sid]
            if verbose:
                print(f"      [DEDUP] Merged '{entry['name']}' ({sid}) → {canonical}")
        else:
            names_seen[norm_name] = sid

    if merge_map:
        for seg in tag_segments:
            old_id = seg.get("skill_id", "")
            if old_id in merge_map:
                seg["skill_id"] = merge_map[old_id]

    sub_episodes: List[SubTask_Experience] = []
    if episodes:
        sub_episodes = _build_sub_episodes_from_tags(
            tag_segments, episodes, game_name, outcome_length=outcome_length,
        )
        if verbose:
            print(f"    Built {len(sub_episodes)} SubTask_Experience objects from tag segments")

    return tag_segments, skill_catalog, sub_episodes


# ═══════════════════════════════════════════════════════════════════════
# Core: Skill extraction for one game (GRPO pipeline with I/O recording)
# ═══════════════════════════════════════════════════════════════════════

def populate_skill_protocols(
    agent: SkillBankAgent,
    skill_catalog: Dict[str, Dict[str, Any]],
    episodes_data: List[Dict[str, Any]],
    game_name: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> int:
    """Fill empty protocols on skills in the bank using Qwen3-8B."""
    from skill_agents_grpo.stage3_mvp.schemas import Protocol

    updated = 0
    bank = agent.bank

    for sid in list(bank.skill_ids):
        skill = bank.get_skill(sid)
        if skill is None or skill.retired:
            continue
        if skill.protocol.steps:
            continue

        contract = skill.contract
        if contract is None:
            continue

        cat_entry = skill_catalog.get(sid, {})
        description = skill.strategic_description or cat_entry.get("description", "")
        name = skill.name or cat_entry.get("name", sid)

        sample_intentions: List[str] = []
        sample_states: List[str] = []

        target_tag = cat_entry.get("tag", "").upper()
        for ep_data in episodes_data:
            for exp in ep_data.get("experiences", []):
                sk = exp.get("skills")
                if sk and isinstance(sk, dict) and sk.get("skill_id") == sid:
                    intent = exp.get("intentions", "")
                    if intent and len(sample_intentions) < 5:
                        sample_intentions.append(intent)
                    ss = exp.get("summary_state", "")
                    if ss and len(sample_states) < 5:
                        sample_states.append(str(ss)[:150])

        if not sample_intentions and target_tag:
            for ep_data in episodes_data:
                for exp in ep_data.get("experiences", []):
                    intent = exp.get("intentions", "")
                    if not intent:
                        continue
                    m = _TAG_RE.match(intent.strip())
                    exp_tag = m.group(1).upper() if m else ""
                    if exp_tag == target_tag or _TAG_ALIASES.get(exp_tag) == target_tag:
                        if len(sample_intentions) < 5:
                            sample_intentions.append(intent)
                        ss = exp.get("summary_state", "")
                        if ss and len(sample_states) < 5:
                            sample_states.append(str(ss)[:150])

        if not sample_intentions:
            tag = target_tag or "EXECUTE"
            sample_intentions = [f"[{tag}] {description[:60]}"]

        proto_dict = generate_skill_protocol(
            sid, name, description, contract, game_name,
            sample_intentions, sample_states, model=model,
        )

        total_steps = cat_entry.get("total_steps", 0)
        n_inst = cat_entry.get("n_instances", contract.n_instances or 1)
        avg_duration = max(1, total_steps // n_inst) if n_inst and total_steps else 10

        protocol = Protocol(
            preconditions=proto_dict["preconditions"],
            steps=proto_dict["steps"],
            success_criteria=proto_dict["success_criteria"],
            abort_criteria=proto_dict["abort_criteria"],
            expected_duration=avg_duration,
        )

        skill.protocol = protocol
        bank.add_or_update_skill(skill)
        updated += 1

        if verbose:
            print(f"      Protocol for {sid}: {len(protocol.steps)} steps, "
                  f"{len(protocol.preconditions)} preconditions")

    return updated


def extract_skills_for_game(
    episodes_data: List[Dict[str, Any]],
    game_name: str,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    resegment: bool = False,
    checkpoint: Optional[ExtractionCheckpoint] = None,
    resume_from_episode: int = 0,
) -> Tuple[SkillBankAgent, Dict[str, Dict[str, Any]], List[SubTask_Experience], StageIOLog]:
    """Run the full skill_agents_grpo SkillBankAgent pipeline on labeled episodes
    for one game, recording I/O at every stage boundary.

    Returns (agent, skill_catalog, sub_episodes, io_log).
    """
    io_log = StageIOLog(game_name, model)
    _reset_llm_call_log()
    from skill_agents_grpo.infer_segmentation.llm_teacher import reset_teacher_io_records
    reset_teacher_io_records()
    from skill_agents_grpo.coldstart_io import reset as _reset_coldstart
    _reset_coldstart()
    game_llm_calls: List[Dict[str, Any]] = []

    _agent_ref_for_snapshot: List = []

    def _on_record_finish(rec: StageIORecord) -> None:
        if _agent_ref_for_snapshot:
            rec.record_output("bank_skills_after", _bank_snapshot(_agent_ref_for_snapshot[0]))

    io_log._on_finish = _on_record_finish

    snapshots_dir = output_dir / "episode_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_episode(label: str, agent_ref, io_log_ref: StageIOLog,
                          llm_calls: List[Dict[str, Any]]) -> None:
        """Save a point-in-time copy of the I/O log, skill bank, and LLM calls."""
        snap_dir = snapshots_dir / label
        snap_dir.mkdir(parents=True, exist_ok=True)
        io_log_ref.save(snap_dir / "stage_io_log.json")
        if agent_ref is not None:
            agent_ref.bank.save(str(snap_dir / "skill_bank.jsonl"))
        if llm_calls:
            with open(snap_dir / "llm_calls.json", "w", encoding="utf-8") as f:
                json.dump({
                    "game": game_name, "model": model,
                    "snapshot": label,
                    "n_calls": len(llm_calls),
                    "calls": llm_calls,
                }, f, indent=2, ensure_ascii=False, default=str)
        print(f"    [Snapshot] {label} → {snap_dir}")

    def _bank_snapshot(agent_ref) -> List[Dict[str, Any]]:
        """Lightweight snapshot of every skill in the bank."""
        snap: List[Dict[str, Any]] = []
        for sid in sorted(agent_ref.skill_ids):
            skill = agent_ref.bank.get_skill(sid)
            if skill is None:
                snap.append({"skill_id": sid, "missing": True})
                continue
            contract = skill.contract
            proto = skill.protocol
            snap.append({
                "skill_id": sid,
                "name": skill.name or sid,
                "retired": bool(skill.retired),
                "n_instances": contract.n_instances if contract else 0,
                "version": contract.version if contract else 0,
                "n_sub_episodes": len(skill.sub_episodes) if skill.sub_episodes else 0,
                "has_protocol": bool(proto and proto.steps),
                "n_protocol_steps": len(proto.steps) if proto and proto.steps else 0,
                "tags": skill.tags[:3] if skill.tags else [],
            })
        return snap

    bank_path = str(output_dir / "skill_bank.jsonl")

    config = PipelineConfig(
        bank_path=bank_path,
        env_name="llm",
        game_name=game_name,
        merge_radius=3,
        extractor_model=model,
        segmentation_method="dp",
        preference_iterations=1,
        new_skill_penalty=2.0,
        eff_freq=0.5,
        min_instances_per_skill=1,
        start_end_window=3,
        new_pool_min_cluster_size=1,
        new_pool_min_consistency=0.3,
        new_pool_min_distinctiveness=0.15,
        min_new_cluster_size=1,
        llm_model=model,
        report_dir=str(output_dir / "reports"),
    )

    agent = SkillBankAgent(config=config)
    _agent_ref_for_snapshot.append(agent)

    # Load existing skill bank if present (incremental update)
    existing_bank_path = output_dir / "skill_bank.jsonl"
    n_loaded = 0
    if existing_bank_path.exists():
        try:
            agent.load()
            n_loaded = len(agent.skill_ids)
            n_unretired = 0
            for sid in list(agent.skill_ids):
                skill = agent.bank.get_skill(sid)
                if skill is not None and skill.retired:
                    skill.retired = False
                    agent.bank.add_or_update_skill(skill)
                    n_unretired += 1
            if n_loaded > 0:
                msg = f"    Loaded existing skill bank: {n_loaded} skill(s) from {existing_bank_path}"
                if n_unretired > 0:
                    msg += f" ({n_unretired} un-retired for re-evaluation)"
                print(msg)
        except Exception as exc:
            print(f"    [WARN] Failed to load existing bank: {exc}")
            n_loaded = 0

    load_rec = io_log.new_record("load_existing_bank")
    load_rec.start()
    load_rec.record_input("bank_path", str(existing_bank_path))
    load_rec.record_input("bank_exists", existing_bank_path.exists())
    load_rec.record_output("n_skills_loaded", n_loaded)
    load_rec.record_output("loaded_skill_ids", list(agent.skill_ids))
    load_rec.finish()

    # Convert episodes
    episodes: List[Episode] = []
    for ep_data in episodes_data:
        try:
            episodes.append(dict_to_episode(ep_data))
        except Exception as exc:
            print(f"    [WARN] Failed to convert episode: {exc}")

    if not episodes:
        print(f"    [WARN] No episodes to segment for {game_name}")
        return agent, {}, [], io_log

    # ══════════════════════════════════════════════════════════════
    # STAGE 1+2: Boundary proposal + skill-sequence decoding
    # ══════════════════════════════════════════════════════════════
    all_sub_episodes: List[SubTask_Experience] = []
    skipped = 0
    if resume_from_episode > 0:
        print(f"    [Stage 1+2] Resuming from episode {resume_from_episode} "
              f"({resume_from_episode} already done, "
              f"{len(episodes) - resume_from_episode} remaining) ...")
    else:
        print(f"    [Stage 1+2] Segmenting {len(episodes)} episode(s) "
              f"via SkillBankAgent.segment_episode ...")

    for i, ep in enumerate(episodes):
        if i < resume_from_episode:
            skipped += 1
            rec = io_log.new_record(f"stage_1_2_episode_{i}")
            rec.start()
            rec.record_input("episode_index", i)
            rec.record_input("skipped", True)
            rec.record_output("resumed_from_checkpoint", True)
            rec.finish()
            continue

        rec = io_log.new_record(f"stage_1_2_episode_{i}")
        rec.start()
        rec.record_input("episode_index", i)
        rec.record_input("n_experiences", len(ep.experiences))
        rec.record_input("episode_task", ep.task)
        rec.record_input("env_name", "llm")
        rec.record_input("skill_names_in_bank", list(agent.skill_ids))

        try:
            result, ep_sub_episodes = agent.segment_episode(ep, env_name="llm")
            all_sub_episodes.extend(ep_sub_episodes)

            n_segs = len(result.segments) if hasattr(result, "segments") else 0
            rec.record_output("n_segments", n_segs)
            rec.record_output("n_sub_episodes", len(ep_sub_episodes))
            rec.record_output("segment_skills", [
                seg.assigned_skill for seg in result.segments
            ] if hasattr(result, "segments") else [])
            rec.record_output("total_score", getattr(result, "total_score", None))
            rec.record_output("accumulated_all_segments_count", len(agent._all_segments))
            rec.record_output("accumulated_new_pool_count", len(agent._new_pool))

            if verbose:
                print(f"      Episode {i}: {len(ep.experiences)} steps → {n_segs} segments, "
                      f"{len(ep_sub_episodes)} sub-episodes")
        except Exception as exc:
            rec.record_error(exc)
            print(f"      [WARN] Episode {i} segmentation failed: {exc}")
            if verbose:
                traceback.print_exc()
        rec.finish()

        ep_llm = _flush_llm_calls()
        game_llm_calls.extend(ep_llm)
        _snapshot_episode(f"episode_{i}", agent, io_log, game_llm_calls)

        # Per-episode bank management (stage 3 -> stage 4)
        _pe_dir = output_dir / "per_episode_bank_management" / f"episode_{i}"
        _pe_dir.mkdir(parents=True, exist_ok=True)
        _pe_rec: Dict[str, Any] = {
            "episode_index": i,
            "game": game_name,
            "timestamp_start": datetime.now().isoformat(),
            "stage_3_contract_learning": None,
            "stage_4_bank_maintenance": None,
            "skill_bank_snapshot_before": _bank_snapshot(agent),
            "skill_bank_snapshot_after": None,
        }

        if agent._all_segments:
            _s3_in = {
                "n_all_segments": len(agent._all_segments),
                "n_trajectories": len(agent._observations_by_traj),
                "segment_skill_labels": list(set(
                    s.skill_label for s in agent._all_segments)),
                "bank_skill_ids_before": list(agent.skill_ids),
            }
            try:
                _s3_sum = agent.run_contract_learning()
                _pe_rec["stage_3_contract_learning"] = {
                    "inputs": _s3_in,
                    "outputs": {
                        "stage3_summary": _safe_serialize(_s3_sum),
                        "bank_skill_ids_after": list(agent.skill_ids),
                        "n_skills_after": len(agent.skill_ids),
                    },
                    "error": None,
                }
                if verbose:
                    print(f"      [Per-ep S3] Episode {i}: "
                          f"{len(agent.skill_ids)} skill(s) after contract learning")
            except Exception as exc:
                _pe_rec["stage_3_contract_learning"] = {
                    "inputs": _s3_in, "outputs": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if verbose:
                    print(f"      [Per-ep S3] Episode {i} failed: {exc}")

        if agent._all_segments and len(agent.skill_ids) > 0:
            _s4_in = {
                "n_all_segments": len(agent._all_segments),
                "n_skills_before": len(agent.skill_ids),
                "skill_ids_before": list(agent.skill_ids),
            }
            try:
                _maint = agent.run_bank_maintenance()
                _sp = getattr(_maint, "split_results", []) or []
                _mg = getattr(_maint, "merge_results", []) or []
                _rf = getattr(_maint, "refine_results", []) or []
                _pe_rec["stage_4_bank_maintenance"] = {
                    "inputs": _s4_in,
                    "outputs": {
                        "n_splits": len(_sp),
                        "n_merges": len(_mg),
                        "n_refines": len(_rf),
                        "alias_map": _safe_serialize(
                            getattr(_maint, "alias_map", {})),
                        "n_skills_after": len(agent.skill_ids),
                        "skill_ids_after": list(agent.skill_ids),
                        "split_details": [_safe_serialize(s) for s in _sp],
                        "merge_details": [_safe_serialize(m) for m in _mg],
                        "refine_details": [_safe_serialize(r) for r in _rf],
                    },
                    "error": None,
                }
                if verbose:
                    print(f"      [Per-ep S4] Episode {i}: "
                          f"{len(_sp)} splits, {len(_mg)} merges, "
                          f"{len(_rf)} refines")
            except Exception as exc:
                _pe_rec["stage_4_bank_maintenance"] = {
                    "inputs": _s4_in, "outputs": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if verbose:
                    print(f"      [Per-ep S4] Episode {i} failed: {exc}")
        else:
            _skip = ("no segments" if not agent._all_segments
                     else "no skills in bank yet")
            _pe_rec["stage_4_bank_maintenance"] = {
                "skipped": True, "reason": _skip}
            if verbose:
                print(f"      [Per-ep S4] Episode {i}: skipped ({_skip})")

        _pe_rec["skill_bank_snapshot_after"] = _bank_snapshot(agent)
        _pe_rec["timestamp_end"] = datetime.now().isoformat()
        with open(_pe_dir / "bank_management_io.json", "w",
                  encoding="utf-8") as f:
            json.dump(_pe_rec, f, indent=2, ensure_ascii=False, default=str)
        agent.bank.save(str(_pe_dir / "skill_bank.jsonl"))

        # Incrementally flush cold-start I/O records
        from skill_agents_grpo.coldstart_io import flush as _flush_coldstart
        _cs_records = _flush_coldstart()
        if _cs_records:
            _cs_path = output_dir / "coldstart_io_all.jsonl"
            try:
                with open(_cs_path, "a", encoding="utf-8") as _csf:
                    for _csr in _cs_records:
                        _csr["episode_index"] = i
                        _csr["game"] = game_name
                        _csf.write(json.dumps(_csr, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass

        print(f"    [Per-ep bank mgmt] episode_{i} → {_pe_dir}")

        if checkpoint is not None:
            checkpoint.mark_episode_done(game_name, i, stage="stage_1_2")

    # ══════════════════════════════════════════════════════════════
    # STAGE 3: Contract learning / verify / refine
    # ══════════════════════════════════════════════════════════════
    if agent._all_segments:
        rec = io_log.new_record("stage_3_contract_learning")
        rec.start()
        rec.record_input("n_all_segments", len(agent._all_segments))
        rec.record_input("n_trajectories", len(agent._observations_by_traj))
        rec.record_input("segment_skill_labels", list(set(
            s.skill_label for s in agent._all_segments
        )))
        rec.record_input("n_new_segments", sum(
            1 for s in agent._all_segments
            if s.skill_label in ("__NEW__", "NEW")
        ))
        rec.record_input("bank_skill_ids_before", list(agent.skill_ids))

        try:
            s3_summary = agent.run_contract_learning()
            rec.record_output("stage3_summary", s3_summary)
            rec.record_output("bank_skill_ids_after", list(agent.skill_ids))
            rec.record_output("n_skills_after", len(agent.skill_ids))
        except Exception as exc:
            rec.record_error(exc)
            if verbose:
                print(f"      [WARN] Stage 3 contract learning failed: {exc}")
        rec.finish()
        if checkpoint is not None:
            checkpoint.mark_stage(game_name, "stage_3")

    # ══════════════════════════════════════════════════════════════
    # LINK SUB-EPISODES (pre quality check)
    # ══════════════════════════════════════════════════════════════
    _pre_qc_linked = 0
    if all_sub_episodes and len(agent.skill_ids) > 0:
        pre_link_rec = io_log.new_record("pre_qc_link_sub_episodes")
        pre_link_rec.start()
        pre_link_rec.record_input("n_sub_episodes", len(all_sub_episodes))
        pre_link_rec.record_input("n_skills", len(agent.skill_ids))
        pre_link_rec.record_input("skill_ids", list(agent.skill_ids))
        sub_task_labels = {}
        for se in all_sub_episodes:
            lbl = getattr(se, "sub_task", None) or ""
            sub_task_labels[lbl] = sub_task_labels.get(lbl, 0) + 1
        pre_link_rec.record_input("sub_task_label_distribution", sub_task_labels)
        try:
            _pre_qc_linked = _link_sub_episodes_to_skills(
                agent, all_sub_episodes, verbose=verbose,
            )
            pre_link_rec.record_output("n_linked", _pre_qc_linked)
            pre_link_rec.record_output("bank_skills_after", _bank_snapshot(agent))
            print(f"    [Pre-QC link] Linked {_pre_qc_linked} sub-episode ref(s) "
                  f"across {len(agent.skill_ids)} skill(s)")
        except Exception as exc:
            pre_link_rec.record_error(exc)
            print(f"    [WARN] Pre-QC sub-episode linking failed: {exc}")
            traceback.print_exc()
        pre_link_rec.finish()

    # ══════════════════════════════════════════════════════════════
    # STAGE 4.5: Sub-episode quality check
    # ══════════════════════════════════════════════════════════════
    if len(agent.skill_ids) > 0:
        from skill_agents_grpo.quality.sub_episode_evaluator import (
            run_quality_check as _run_qc_single,
        )

        total_dropped = 0
        total_retired = 0
        n_checked = 0
        print(f"    [Stage 4.5] Quality-checking {len(agent.skill_ids)} skill(s) one-by-one ...")

        for sid in list(agent.skill_ids):
            skill = agent.bank.get_skill(sid)
            if skill is None or skill.retired:
                rec = io_log.new_record(f"stage_4_5_quality_check/{sid}")
                rec.start()
                rec.record_input("skill_id", sid)
                rec.record_input("skipped", True)
                rec.record_input("reason", "skill is None" if skill is None else "retired")
                rec.record_output("action", "skipped")
                rec.finish()
                continue

            rec = io_log.new_record(f"stage_4_5_quality_check/{sid}")
            rec.start()

            sub_eps_before = skill.sub_episodes or []
            rec.record_input("skill_id", sid)
            rec.record_input("skill_name", skill.name)
            rec.record_input("n_sub_episodes_before", len(sub_eps_before))
            rec.record_input("sub_episodes_detail", [
                {
                    "seg_start": se.seg_start,
                    "seg_end": se.seg_end,
                    "outcome": se.outcome,
                    "cumulative_reward": se.cumulative_reward,
                    "quality_score_before": se.quality_score,
                    "intention_tags": se.intention_tags[:5] if se.intention_tags else [],
                    "summary": se.summary[:120] if se.summary else "",
                }
                for se in sub_eps_before
            ])

            try:
                qc_result = _run_qc_single(skill)
                agent.bank.add_or_update_skill(skill)

                sub_eps_after = skill.sub_episodes or []
                rec.record_output("before_count", qc_result["before_count"])
                rec.record_output("dropped", qc_result["dropped"])
                rec.record_output("after_count", qc_result["after_count"])
                rec.record_output("needs_protocol_update", qc_result["needs_protocol_update"])
                rec.record_output("retired", qc_result["retired"])
                rec.record_output("sub_episodes_after", [
                    {
                        "seg_start": se.seg_start,
                        "seg_end": se.seg_end,
                        "outcome": se.outcome,
                        "quality_score": se.quality_score,
                    }
                    for se in sub_eps_after
                ])

                total_dropped += qc_result["dropped"]
                if qc_result["retired"]:
                    total_retired += 1
                n_checked += 1

                if verbose:
                    status = "RETIRED" if qc_result["retired"] else "ok"
                    print(f"      {sid}: {qc_result['before_count']}→{qc_result['after_count']} "
                          f"sub-eps (dropped {qc_result['dropped']}) [{status}]")
            except Exception as exc:
                rec.record_error(exc)
                if verbose:
                    print(f"      [WARN] Quality check for {sid} failed: {exc}")
            rec.finish()

        if n_checked > 0:
            print(f"    Stage 4.5 complete: {n_checked} skills checked, "
                  f"{total_dropped} sub-eps dropped, {total_retired} retired")
        if checkpoint is not None:
            checkpoint.mark_stage(game_name, "stage_4_5")

    # ══════════════════════════════════════════════════════════════
    # STAGE 4: Bank maintenance
    # ══════════════════════════════════════════════════════════════
    if agent._all_segments and len(agent.skill_ids) > 0:
        overview_rec = io_log.new_record("stage_4_bank_maintenance")
        overview_rec.start()
        overview_rec.record_input("n_all_segments", len(agent._all_segments))
        overview_rec.record_input("n_skills_before", len(agent.skill_ids))
        overview_rec.record_input("skill_ids_before", list(agent.skill_ids))

        try:
            maint_result = agent.run_bank_maintenance()
            split_results = maint_result.split_results if hasattr(maint_result, "split_results") else []
            merge_results = maint_result.merge_results if hasattr(maint_result, "merge_results") else []
            refine_results = maint_result.refine_results if hasattr(maint_result, "refine_results") else []

            overview_rec.record_output("n_splits", len(split_results))
            overview_rec.record_output("n_merges", len(merge_results))
            overview_rec.record_output("n_refines", len(refine_results))
            overview_rec.record_output("alias_map", getattr(maint_result, "alias_map", {}))
            overview_rec.record_output("n_skills_after", len(agent.skill_ids))
            overview_rec.record_output("skill_ids_after", list(agent.skill_ids))
            overview_rec.finish()

            for sr in split_results:
                split_rec = io_log.new_record(f"stage_4_split/{sr.parent_id}")
                split_rec.start()
                split_rec.record_input("parent_id", sr.parent_id)
                split_rec.record_output("accepted", sr.accepted)
                split_rec.record_output("reason", sr.reason)
                split_rec.record_output("children", [
                    {
                        "skill_id": c.skill_id,
                        "parent_id": c.parent_id,
                        "n_instance_seg_ids": len(c.instance_seg_ids),
                        "pass_rate": c.report.overall_pass_rate if c.report else None,
                    }
                    for c in sr.children
                ] if sr.children else [])
                split_rec.finish()

            for mr in merge_results:
                merge_rec = io_log.new_record(f"stage_4_merge/{mr.canonical_id}")
                merge_rec.start()
                merge_rec.record_input("canonical_id", mr.canonical_id)
                merge_rec.record_input("merged_ids", mr.merged_ids)
                merge_rec.record_output("accepted", mr.accepted)
                merge_rec.record_output("reason", mr.reason)
                merge_rec.record_output("alias_map", mr.alias_map)
                merge_rec.record_output("pass_rate", mr.report.overall_pass_rate if mr.report else None)
                merge_rec.finish()

            for rr in refine_results:
                refine_rec = io_log.new_record(f"stage_4_refine/{rr.skill_id}")
                refine_rec.start()
                refine_rec.record_input("skill_id", rr.skill_id)
                refine_rec.record_output("reason", rr.reason)
                refine_rec.record_output("dropped_literals", rr.dropped_literals)
                refine_rec.record_output("added_literals", rr.added_literals)
                refine_rec.record_output("has_new_contract", rr.new_contract is not None)
                refine_rec.finish()

            if verbose:
                print(f"      Stage 4 bank maintenance: {len(split_results)} splits, "
                      f"{len(merge_results)} merges, {len(refine_results)} refines")
        except Exception as exc:
            overview_rec.record_error(exc)
            overview_rec.finish()
            if verbose:
                print(f"      [WARN] Stage 4 bank maintenance failed: {exc}")
        if checkpoint is not None:
            checkpoint.mark_stage(game_name, "stage_4")

    # ══════════════════════════════════════════════════════════════
    # PROTO-SKILL FORMATION / VERIFICATION / PROMOTION
    # ══════════════════════════════════════════════════════════════
    rec = io_log.new_record("proto_skill_formation")
    rec.start()
    rec.record_input("new_pool_size", agent._new_pool_mgr.size)
    rec.record_input("new_pool_legacy_size", len(agent._new_pool))
    rec.record_input("existing_bank_skill_ids", list(agent.skill_ids))

    try:
        n_proto = agent.form_proto_skills()
        rec.record_output("n_proto_skills_created", n_proto)
        rec.record_output("proto_skill_ids", list(agent._proto_mgr.proto_ids) if hasattr(agent._proto_mgr, "proto_ids") else [])
        if verbose and n_proto > 0:
            print(f"      Formed {n_proto} proto-skill(s)")
    except Exception as exc:
        n_proto = 0
        rec.record_error(exc)
        if verbose:
            print(f"      [WARN] Proto-skill formation failed: {exc}")
    rec.finish()

    if n_proto > 0:
        rec = io_log.new_record("proto_skill_verification")
        rec.start()
        rec.record_input("n_proto_skills", n_proto)
        rec.record_input("proto_skill_ids", list(agent._proto_mgr.proto_ids) if hasattr(agent._proto_mgr, "proto_ids") else [])

        try:
            n_verified = agent.verify_proto_skills()
            rec.record_output("n_proto_skills_verified", n_verified)
            if verbose and n_verified > 0:
                print(f"      Verified {n_verified} proto-skill(s)")
        except Exception as exc:
            rec.record_error(exc)
            if verbose:
                print(f"      [WARN] Proto-skill verification failed: {exc}")
        rec.finish()

    rec = io_log.new_record("proto_skill_promotion")
    rec.start()
    rec.record_input("bank_skill_ids_before", list(agent.skill_ids))
    rec.record_input("proto_skill_ids", list(agent._proto_mgr.proto_ids) if hasattr(agent._proto_mgr, "proto_ids") else [])

    try:
        n_promoted = agent.promote_proto_skills()
        rec.record_output("n_promoted", n_promoted)
        rec.record_output("bank_skill_ids_after", list(agent.skill_ids))
        if verbose and n_promoted > 0:
            print(f"      Promoted {n_promoted} proto-skill(s) to real skills")
    except Exception as exc:
        rec.record_error(exc)
        if verbose:
            print(f"      [WARN] Proto-skill promotion failed: {exc}")
    rec.finish()

    # ══════════════════════════════════════════════════════════════
    # MATERIALIZE NEW SKILLS
    # ══════════════════════════════════════════════════════════════
    rec = io_log.new_record("materialize_new_skills")
    rec.start()
    rec.record_input("new_pool_mgr_size", agent._new_pool_mgr.size)
    rec.record_input("new_pool_legacy_size", len(agent._new_pool))
    rec.record_input("bank_skill_ids_before", list(agent.skill_ids))
    rec.record_input("min_new_cluster_size", config.min_new_cluster_size)

    try:
        n_materialized = agent.materialize_new_skills()
        rec.record_output("n_materialized", n_materialized)
        rec.record_output("bank_skill_ids_after", list(agent.skill_ids))
        if verbose and n_materialized > 0:
            print(f"      Materialized {n_materialized} new skill(s)")
    except Exception as exc:
        rec.record_error(exc)
        if verbose:
            print(f"      [WARN] Materialize new skills failed: {exc}")
    rec.finish()

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: Distill execution hints
    # ══════════════════════════════════════════════════════════════
    if len(agent.skill_ids) > 0:
        rec = io_log.new_record("phase_5_execution_hints")
        rec.start()
        rec.record_input("n_skills", len(agent.skill_ids))
        skills_with_hints_before = sum(
            1 for sid in agent.skill_ids
            if (agent.bank.get_skill(sid) or type("", (), {"execution_hint": None})()).execution_hint is not None
        )
        rec.record_input("skills_with_hints_before", skills_with_hints_before)

        try:
            n_hints = agent.distill_execution_hints()
            rec.record_output("n_hints_distilled", n_hints)
            skills_with_hints_after = sum(
                1 for sid in agent.skill_ids
                if (agent.bank.get_skill(sid) or type("", (), {"execution_hint": None})()).execution_hint is not None
            )
            rec.record_output("skills_with_hints_after", skills_with_hints_after)
            if verbose and n_hints > 0:
                print(f"      Distilled {n_hints} execution hint(s)")
        except Exception as exc:
            rec.record_error(exc)
            if verbose:
                print(f"      [WARN] Execution hint distillation failed: {exc}")
        rec.finish()

    # ══════════════════════════════════════════════════════════════
    # PROTOCOL UPDATE (via LLM synthesis)
    # ══════════════════════════════════════════════════════════════
    if len(agent.skill_ids) > 0:
        rec = io_log.new_record("protocol_update_llm")
        rec.start()
        skills_with_protocol_before = sum(
            1 for sid in agent.skill_ids
            if (agent.bank.get_skill(sid) or type("", (), {"protocol": type("", (), {"steps": []})()})()).protocol.steps
        )
        rec.record_input("n_skills", len(agent.skill_ids))
        rec.record_input("skills_with_protocol_before", skills_with_protocol_before)

        try:
            n_proto_updated = agent.update_protocols()
            rec.record_output("n_protocols_updated", n_proto_updated)
            if verbose and n_proto_updated > 0:
                print(f"      Updated {n_proto_updated} protocol(s) via LLM synthesis")
        except Exception as exc:
            rec.record_error(exc)
            if verbose:
                print(f"      [WARN] Protocol update failed: {exc}")
        rec.finish()

    # ══════════════════════════════════════════════════════════════
    # SKILL EVALUATION
    # ══════════════════════════════════════════════════════════════
    if len(agent.skill_ids) > 0:
        rec = io_log.new_record("skill_evaluation")
        rec.start()
        rec.record_input("n_skills", len(agent.skill_ids))
        rec.record_input("skill_ids", list(agent.skill_ids))
        rec.record_input("n_all_segments", len(agent._all_segments))

        try:
            eval_summary = agent.run_evaluation()
            n_eval = len(eval_summary.skill_reports) if hasattr(eval_summary, "skill_reports") else 0
            rec.record_output("n_skills_evaluated", n_eval)
            rec.record_output("evaluation_summary", eval_summary)
            if verbose:
                print(f"      Evaluation: {n_eval} skill(s) evaluated")
        except Exception as exc:
            rec.record_error(exc)
            if verbose:
                print(f"      [WARN] Skill evaluation failed: {exc}")
        rec.finish()

    pipeline_skills = len(agent.skill_ids)
    new_pipeline_skills = pipeline_skills - n_loaded
    if pipeline_skills > 0:
        print(f"    SkillBankAgent (GRPO) has {pipeline_skills} skill(s) "
              f"({n_loaded} loaded, {new_pipeline_skills} new from pipeline)")
    else:
        print(f"    SkillBankAgent (GRPO) produced 0 new skills — "
              f"running intention-based segmentation for discovery")

    # Fallback: intention-based segmentation
    all_new_are_protos = (
        n_loaded == 0
        and len(agent.skill_ids) > 0
        and all(sid.startswith("proto_") for sid in agent.skill_ids)
    )
    use_intention_fallback = new_pipeline_skills == 0 or all_new_are_protos
    skill_catalog: Dict[str, Dict[str, Any]] = {}

    for sid in list(agent.skill_ids):
        skill = agent.bank.get_skill(sid)
        if skill is None or skill.retired:
            continue
        contract = skill.contract
        if contract is None:
            continue
        skill_catalog[sid] = {
            "skill_id": sid,
            "name": skill.name or sid,
            "summary": skill.strategic_description or "",
            "description": skill.strategic_description or "",
            "tag": skill.tags[0] if skill.tags else "",
            "eff_add": sorted(contract.eff_add) if contract.eff_add else [],
            "eff_del": sorted(contract.eff_del) if contract.eff_del else [],
            "eff_event": sorted(contract.eff_event) if contract.eff_event else [],
            "n_instances": contract.n_instances,
            "version": contract.version,
        }

    if use_intention_fallback:
        if all_new_are_protos:
            for sid in list(agent.skill_ids):
                if sid.startswith("proto_"):
                    skill = agent.bank.get_skill(sid)
                    if skill is not None:
                        skill.retired = True
                        agent.bank.add_or_update_skill(skill)
                    if verbose:
                        print(f"      Retired undifferentiated proto skill {sid}")

        fb_rec = io_log.new_record("fallback_intention_segmentation")
        fb_rec.start()
        fb_rec.record_input("n_episodes_data", len(episodes_data))
        fb_rec.record_input("game_name", game_name)
        fb_rec.record_input("model", model)
        fb_rec.record_input("existing_skill_ids", list(agent.skill_ids))
        fb_rec.record_input("retired_protos", all_new_are_protos)

        _flush_llm_calls()
        _tag_segments, fb_catalog, fallback_sub_episodes = intention_based_segmentation(
            episodes_data, game_name, episodes=episodes,
            model=model, verbose=verbose,
        )
        all_sub_episodes = fallback_sub_episodes

        fb_llm_calls = _flush_llm_calls()
        game_llm_calls.extend(fb_llm_calls)

        n_new_from_fallback = 0
        for sid, entry in fb_catalog.items():
            existing = agent.bank.get_skill(sid)
            if existing is not None and not existing.retired:
                existing.n_instances = max(existing.n_instances, entry.get("n_instances", 0))
                agent.bank.add_or_update_skill(existing)
                skill_catalog[sid] = entry
                continue

            contract = SkillEffectsContract(
                skill_id=sid,
                name=entry["name"],
                description=entry["description"],
                eff_add=set(entry.get("eff_add", [])),
                eff_del=set(entry.get("eff_del", [])),
                eff_event=set(entry.get("eff_event", [])),
                n_instances=entry.get("n_instances", 0),
            )
            agent.bank.add_or_update(contract)
            skill = agent.bank.get_skill(sid)
            if skill is not None:
                skill.name = entry["name"]
                skill.strategic_description = entry.get("description", "")
                tag = entry.get("tag", "")
                if tag:
                    skill.tags = [tag]
                    skill.expected_tag_pattern = [tag]
                agent.bank.add_or_update_skill(skill)
            skill_catalog[sid] = entry
            n_new_from_fallback += 1

        fb_rec.record_output("n_tag_segments", len(_tag_segments))
        fb_rec.record_output("n_skills_in_catalog", len(fb_catalog))
        fb_rec.record_output("n_new_skills_added", n_new_from_fallback)
        fb_rec.record_output("n_existing_skills_updated", len(fb_catalog) - n_new_from_fallback)
        fb_rec.record_output("n_sub_episodes", len(fallback_sub_episodes))
        fb_rec.record_output("skill_ids", list(fb_catalog.keys()))
        fb_rec.record_output("n_llm_calls", len(fb_llm_calls))
        fb_rec.record_output("llm_call_summary", [
            {"caller": c["caller"], "elapsed_s": c["elapsed_s"],
             "response_chars": c["response_chars"]}
            for c in fb_llm_calls
        ])
        fb_rec.finish()

        # Post-fallback quality check
        if n_new_from_fallback > 0 and all_sub_episodes:
            from skill_agents_grpo.quality.sub_episode_evaluator import (
                run_quality_check as _run_qc_single,
            )
            print(f"    [Stage 4.5 post-fallback] Quality-checking {len(agent.skill_ids)} skill(s) ...")
            for sid in list(agent.skill_ids):
                skill = agent.bank.get_skill(sid)
                if skill is None or skill.retired:
                    rec = io_log.new_record(f"stage_4_5_quality_check/{sid}")
                    rec.start()
                    rec.record_input("skill_id", sid)
                    rec.record_input("skipped", True)
                    rec.record_input("reason", "skill is None" if skill is None else "retired")
                    rec.record_output("action", "skipped")
                    rec.finish()
                    continue
                rec = io_log.new_record(f"stage_4_5_quality_check/{sid}")
                rec.start()
                sub_eps_before = skill.sub_episodes or []
                rec.record_input("skill_id", sid)
                rec.record_input("skill_name", skill.name)
                rec.record_input("n_sub_episodes_before", len(sub_eps_before))
                rec.record_input("sub_episodes_detail", [
                    {
                        "seg_start": se.seg_start, "seg_end": se.seg_end,
                        "outcome": se.outcome, "cumulative_reward": se.cumulative_reward,
                        "quality_score_before": se.quality_score,
                    }
                    for se in sub_eps_before
                ])
                try:
                    qc_result = _run_qc_single(skill)
                    agent.bank.add_or_update_skill(skill)
                    rec.record_output("before_count", qc_result["before_count"])
                    rec.record_output("dropped", qc_result["dropped"])
                    rec.record_output("after_count", qc_result["after_count"])
                    rec.record_output("needs_protocol_update", qc_result["needs_protocol_update"])
                    rec.record_output("retired", qc_result["retired"])
                    if verbose:
                        status = "RETIRED" if qc_result["retired"] else "ok"
                        print(f"      {sid}: {qc_result['before_count']}→{qc_result['after_count']} "
                              f"sub-eps (dropped {qc_result['dropped']}) [{status}]")
                except Exception as exc:
                    rec.record_error(exc)
                rec.finish()

    else:
        cat_rec = io_log.new_record("build_skill_catalog_from_pipeline")
        cat_rec.start()
        cat_rec.record_input("n_pipeline_skills", pipeline_skills)
        cat_rec.record_input("skill_ids", list(agent.skill_ids))
        _flush_llm_calls()

        for sid in agent.skill_ids:
            contract = agent.get_contract(sid)
            if contract is None:
                continue

            sample_intentions: List[str] = []
            sample_states: List[str] = []
            for seg in agent.segments:
                if seg.skill_label != sid:
                    continue
                traj_obs = agent._observations_by_traj.get(seg.traj_id, [])
                for t in range(seg.t_start, min(seg.t_end, len(traj_obs))):
                    obs = traj_obs[t]
                    if obs and len(sample_states) < 5:
                        sample_states.append(str(obs)[:200])
                for ep_data in episodes_data:
                    exps = ep_data.get("experiences", [])
                    for t in range(seg.t_start, min(seg.t_end, len(exps))):
                        intent = exps[t].get("intentions", "")
                        if intent and len(sample_intentions) < 5:
                            sample_intentions.append(intent)

            name, rag_summary = generate_skill_name(
                sid, contract, game_name, sample_intentions, model=model,
            )
            description = generate_skill_description(
                sid, name, contract, game_name, sample_states, model=model,
            )

            contract.name = name
            contract.description = description
            agent.bank.add_or_update(contract)

            skill = agent.bank.get_skill(sid)
            if skill is not None:
                if not skill.name or skill.name == sid:
                    skill.name = name
                if not skill.strategic_description:
                    skill.strategic_description = description
                agent.bank.add_or_update_skill(skill)

            skill_catalog[sid] = {
                "skill_id": sid,
                "name": name,
                "summary": rag_summary,
                "description": description,
                "eff_add": sorted(contract.eff_add) if contract.eff_add else [],
                "eff_del": sorted(contract.eff_del) if contract.eff_del else [],
                "eff_event": sorted(contract.eff_event) if contract.eff_event else [],
                "n_instances": contract.n_instances,
                "version": contract.version,
            }

            if verbose:
                print(f"      Skill {sid}: {name}")
                print(f"        summary: {rag_summary[:80]}...")

        cat_llm_calls = _flush_llm_calls()
        game_llm_calls.extend(cat_llm_calls)
        cat_rec.record_output("n_catalog_entries", len(skill_catalog))
        cat_rec.record_output("catalog_skill_ids", list(skill_catalog.keys()))
        cat_rec.record_output("n_llm_calls", len(cat_llm_calls))
        cat_rec.record_output("llm_call_summary", [
            {"caller": c["caller"], "elapsed_s": c["elapsed_s"],
             "response_chars": c["response_chars"]}
            for c in cat_llm_calls
        ])
        cat_rec.finish()

    if checkpoint is not None:
        checkpoint.mark_stage(game_name, "catalog_built")

    # Optional re-segmentation pass against seeded bank
    if resegment and len(agent.skill_ids) > 0 and episodes:
        print(f"    [Re-segment] Re-segmenting {len(episodes)} episode(s) against "
              f"seeded bank ({len(agent.skill_ids)} skills) ...")
        reseg_agent = SkillBankAgent(config=config)
        reseg_agent.bank = agent.bank

        reseg_rec = io_log.new_record("resegment_pass")
        reseg_rec.start()
        reseg_rec.record_input("n_episodes", len(episodes))
        reseg_rec.record_input("n_seeded_skills", len(agent.skill_ids))
        reseg_rec.record_input("seeded_skill_ids", list(agent.skill_ids))

        reseg_sub_episodes: List[SubTask_Experience] = []
        for i, ep in enumerate(episodes):
            try:
                result, ep_sub = reseg_agent.segment_episode(ep, env_name="llm")
                reseg_sub_episodes.extend(ep_sub)
                n_segs = len(result.segments) if hasattr(result, "segments") else 0
                if verbose:
                    print(f"      Re-seg episode {i}: {len(ep.experiences)} steps → "
                          f"{n_segs} segments, {len(ep_sub)} sub-episodes")
            except Exception as exc:
                print(f"      [WARN] Re-seg episode {i} failed: {exc}")
                if verbose:
                    traceback.print_exc()

        if reseg_agent._all_segments:
            try:
                reseg_agent.run_contract_learning()
            except Exception as exc:
                if verbose:
                    print(f"      [WARN] Re-seg Stage 3 failed: {exc}")

            if len(reseg_agent.skill_ids) > 0:
                try:
                    reseg_agent.run_sub_episode_quality_check()
                except Exception as exc:
                    if verbose:
                        print(f"      [WARN] Re-seg Stage 4.5 failed: {exc}")

            if len(reseg_agent.skill_ids) > 0:
                try:
                    reseg_agent.run_bank_maintenance()
                except Exception as exc:
                    if verbose:
                        print(f"      [WARN] Re-seg Stage 4 failed: {exc}")

            try:
                reseg_agent.form_proto_skills()
                reseg_agent.verify_proto_skills()
                reseg_agent.promote_proto_skills()
            except Exception:
                pass

            try:
                reseg_agent.materialize_new_skills()
            except Exception:
                pass

            if len(reseg_agent.skill_ids) > 0:
                try:
                    reseg_agent.distill_execution_hints()
                except Exception:
                    pass
                try:
                    reseg_agent.update_protocols()
                except Exception:
                    pass
                try:
                    reseg_agent.run_evaluation()
                except Exception:
                    pass

        reseg_rec.record_output("n_reseg_sub_episodes", len(reseg_sub_episodes))
        reseg_rec.record_output("n_reseg_skills", len(reseg_agent.skill_ids))
        reseg_rec.finish()

        if reseg_sub_episodes:
            all_sub_episodes = reseg_sub_episodes
            agent = reseg_agent
            print(f"    Re-segmentation produced {len(reseg_sub_episodes)} sub-episodes, "
                  f"{len(reseg_agent.skill_ids)} skills")

    # Qwen3-8B protocol generation for skills with empty protocols
    if len(agent.skill_ids) > 0:
        proto_gen_rec = io_log.new_record("qwen3_protocol_generation")
        proto_gen_rec.start()
        proto_gen_rec.record_input("n_skills", len(agent.skill_ids))

        print(f"    [Qwen3-8B] Generating protocols for skills with empty protocols ...")
        _flush_llm_calls()
        try:
            n_protos = populate_skill_protocols(
                agent, skill_catalog, episodes_data, game_name,
                model=model, verbose=verbose,
            )
            proto_llm_calls = _flush_llm_calls()
            game_llm_calls.extend(proto_llm_calls)
            proto_gen_rec.record_output("n_protocols_generated", n_protos)
            proto_gen_rec.record_output("n_llm_calls", len(proto_llm_calls))
            proto_gen_rec.record_output("llm_call_summary", [
                {"caller": c["caller"], "elapsed_s": c["elapsed_s"],
                 "response_chars": c["response_chars"]}
                for c in proto_llm_calls
            ])
            if n_protos > 0:
                print(f"    Generated {n_protos} protocol(s) via Qwen3-8B")
            else:
                print(f"    All skills already have protocols")
        except Exception as exc:
            proto_llm_calls = _flush_llm_calls()
            game_llm_calls.extend(proto_llm_calls)
            proto_gen_rec.record_error(exc)
            print(f"    [WARN] Protocol generation failed: {exc}")
            if verbose:
                traceback.print_exc()
        proto_gen_rec.finish()
        if checkpoint is not None:
            checkpoint.mark_stage(game_name, "protocols_done")

    # Final sub-episode linking
    if all_sub_episodes and len(agent.skill_ids) > 0:
        link_rec = io_log.new_record("link_sub_episodes")
        link_rec.start()
        link_rec.record_input("n_sub_episodes", len(all_sub_episodes))
        link_rec.record_input("n_skills", len(agent.skill_ids))

        try:
            n_linked = _link_sub_episodes_to_skills(agent, all_sub_episodes, verbose=verbose)
            link_rec.record_output("n_linked", n_linked)
            if n_linked > 0:
                print(f"    Linked {n_linked} sub-episode ref(s) to skills")
        except Exception as exc:
            link_rec.record_error(exc)
            print(f"    [WARN] Sub-episode linking failed: {exc}")
            if verbose:
                traceback.print_exc()

        min_viable = 2
        n_unretired = 0
        for sid in list(agent.skill_ids):
            skill = agent.bank.get_skill(sid)
            if skill is None or not skill.retired:
                continue
            if len(skill.sub_episodes) >= min_viable:
                skill.retired = False
                agent.bank.add_or_update_skill(skill)
                n_unretired += 1
                if verbose:
                    print(f"      Un-retired {sid} ({len(skill.sub_episodes)} sub-episodes)")
        if n_unretired > 0:
            print(f"    Un-retired {n_unretired} skill(s) after final sub-episode linking")
            link_rec.record_output("n_unretired", n_unretired)

        link_rec.record_output("bank_skills_after", _bank_snapshot(agent))
        link_rec.finish()

    # Late protocol update for un-retired skills
    if len(agent.skill_ids) > 0:
        try:
            n_late_protos = agent.update_protocols()
            if n_late_protos > 0:
                print(f"    Late protocol update: {n_late_protos} protocol(s) synthesized")
        except Exception as exc:
            print(f"    [WARN] Late protocol update failed: {exc}")

    # Name unnamed skills
    unnamed_sids = [
        sid for sid in agent.skill_ids
        if not (agent.bank.get_skill(sid) or type("", (), {"name": ""})()).name
        or (agent.bank.get_skill(sid) or type("", (), {"name": ""})()).name == sid
    ]
    if unnamed_sids:
        _flush_llm_calls()
        print(f"    Naming {len(unnamed_sids)} unnamed skill(s) ...")
        for sid in unnamed_sids:
            contract = agent.get_contract(sid)
            if contract is None:
                continue
            sample_intentions: List[str] = []
            for seg in agent.segments:
                if seg.skill_label != sid:
                    continue
                for ep_data in episodes_data:
                    exps = ep_data.get("experiences", [])
                    for t in range(seg.t_start, min(seg.t_end, len(exps))):
                        intent = exps[t].get("intentions", "")
                        if intent and len(sample_intentions) < 5:
                            sample_intentions.append(intent)
                if len(sample_intentions) >= 5:
                    break
            try:
                name, rag_summary = generate_skill_name(
                    sid, contract, game_name, sample_intentions, model=model,
                )
                description = generate_skill_description(
                    sid, name, contract, game_name, [], model=model,
                )
                skill = agent.bank.get_skill(sid)
                if skill is not None:
                    skill.name = name
                    skill.strategic_description = description
                    agent.bank.add_or_update_skill(skill)
                    if verbose:
                        print(f"      Named {sid} → {name}")
                if sid not in skill_catalog:
                    skill_catalog[sid] = {
                        "skill_id": sid,
                        "name": name,
                        "summary": rag_summary,
                        "description": description,
                        "eff_add": sorted(contract.eff_add) if contract.eff_add else [],
                        "eff_del": sorted(contract.eff_del) if contract.eff_del else [],
                        "eff_event": sorted(contract.eff_event) if contract.eff_event else [],
                        "n_instances": contract.n_instances,
                        "version": contract.version,
                    }
            except Exception as exc:
                print(f"      [WARN] Failed to name skill {sid}: {exc}")
        late_naming_calls = _flush_llm_calls()
        game_llm_calls.extend(late_naming_calls)
        print(f"    Named {len(unnamed_sids)} skill(s)")

    # Final snapshot
    _snapshot_episode("final", agent, io_log, game_llm_calls)

    # Persist skill bank
    try:
        agent.save()
    except Exception as exc:
        print(f"    [WARN] Failed to save skill bank: {exc}")

    # Persist SubTask_Experience objects
    if all_sub_episodes:
        sub_ep_path = output_dir / "sub_episodes.json"
        try:
            sub_ep_data = {
                "game": game_name,
                "model": model,
                "pipeline": "skill_agents_grpo",
                "timestamp": datetime.now().isoformat(),
                "n_sub_episodes": len(all_sub_episodes),
                "sub_episodes": [se.to_dict() for se in all_sub_episodes],
            }
            with open(sub_ep_path, "w", encoding="utf-8") as f:
                json.dump(sub_ep_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"    Sub-episodes ({len(all_sub_episodes)}) → {sub_ep_path}")
        except Exception as exc:
            print(f"    [WARN] Failed to save sub_episodes: {exc}")

    # Save stage I/O log
    io_log_path = output_dir / "stage_io_log.json"
    io_log.save(io_log_path)
    print(f"    Stage I/O log → {io_log_path}")

    # Flush LLM calls and save log
    remaining = _flush_llm_calls()
    game_llm_calls.extend(remaining)

    if game_llm_calls:
        llm_log_path = output_dir / "llm_calls_log.json"
        try:
            llm_log_data = {
                "game": game_name,
                "model": model,
                "pipeline": "skill_agents_grpo",
                "timestamp": datetime.now().isoformat(),
                "n_calls": len(game_llm_calls),
                "total_prompt_chars": sum(c.get("prompt_chars", 0) for c in game_llm_calls),
                "total_response_chars": sum(c.get("response_chars", 0) for c in game_llm_calls),
                "total_elapsed_s": round(sum(c.get("elapsed_s", 0) for c in game_llm_calls), 3),
                "calls": game_llm_calls,
            }
            with open(llm_log_path, "w", encoding="utf-8") as f:
                json.dump(llm_log_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"    LLM call log  → {llm_log_path} ({len(game_llm_calls)} calls)")
        except Exception as exc:
            print(f"    [WARN] Failed to save LLM call log: {exc}")

    # Flush teacher I/O (cold-start data)
    from skill_agents_grpo.infer_segmentation.llm_teacher import flush_teacher_io_records

    teacher_records = flush_teacher_io_records()
    if teacher_records:
        teacher_io_path = output_dir / "teacher_io_coldstart.jsonl"
        try:
            with open(teacher_io_path, "w", encoding="utf-8") as f:
                for rec in teacher_records:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

            by_fn = {}
            for rec in teacher_records:
                fn = rec.get("function", "unknown")
                by_fn[fn] = by_fn.get(fn, 0) + 1
            print(f"    Teacher I/O   → {teacher_io_path} ({len(teacher_records)} records: {by_fn})")
        except Exception as exc:
            print(f"    [WARN] Failed to save teacher I/O: {exc}")

    # Flush remaining cold-start I/O
    from skill_agents_grpo.coldstart_io import flush as _flush_coldstart_final

    remaining_cs = _flush_coldstart_final()
    if remaining_cs:
        cs_path = output_dir / "coldstart_io_all.jsonl"
        try:
            with open(cs_path, "a", encoding="utf-8") as f:
                for rec in remaining_cs:
                    rec["game"] = game_name
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            print(f"    [WARN] Failed to save cold-start I/O: {exc}")

    cs_path = output_dir / "coldstart_io_all.jsonl"
    if cs_path.exists():
        try:
            n_cs = sum(1 for _ in open(cs_path, "r", encoding="utf-8"))
            by_mod: Dict[str, int] = {}
            with open(cs_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        mod = rec.get("module", "unknown")
                        by_mod[mod] = by_mod.get(mod, 0) + 1
                    except Exception:
                        pass
            print(f"    Cold-start IO → {cs_path} ({n_cs} records: {by_mod})")
        except Exception:
            pass

    return agent, skill_catalog, all_sub_episodes, io_log


# ═══════════════════════════════════════════════════════════════════════
# Annotate episodes with extracted skills
# ═══════════════════════════════════════════════════════════════════════

def annotate_episodes_with_skills(
    episodes_data: List[Dict[str, Any]],
    agent: SkillBankAgent,
    skill_catalog: Dict[str, Dict[str, Any]],
    verbose: bool = False,
) -> None:
    """Populate the ``skills`` field on each experience in each episode."""
    step_to_skill: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for seg in agent.segments:
        if seg.skill_label in ("__NEW__", "NEW"):
            continue
        catalog_entry = skill_catalog.get(seg.skill_label, {})
        if not catalog_entry:
            continue
        skill_info = {
            "skill_id": seg.skill_label,
            "skill_name": catalog_entry.get("name", seg.skill_label),
            "skill_summary": catalog_entry.get("summary", ""),
            "description": catalog_entry.get("description", ""),
            "segment_start": seg.t_start,
            "segment_end": seg.t_end,
            "eff_add": catalog_entry.get("eff_add", []),
            "eff_del": catalog_entry.get("eff_del", []),
            "eff_event": catalog_entry.get("eff_event", []),
        }
        for t in range(seg.t_start, seg.t_end):
            step_to_skill[seg.traj_id][t] = skill_info

    has_intention_skills = any("tag" in v for v in skill_catalog.values())

    if has_intention_skills:
        tag_to_info: Dict[str, Dict[str, Any]] = {}
        for sid, entry in skill_catalog.items():
            tag = entry.get("tag")
            if tag:
                tag_to_info[tag] = {
                    "skill_id": sid,
                    "skill_name": entry.get("name", sid),
                    "skill_summary": entry.get("summary", ""),
                    "description": entry.get("description", ""),
                    "eff_add": entry.get("eff_add", []),
                    "eff_del": entry.get("eff_del", []),
                    "eff_event": entry.get("eff_event", []),
                }

        for ep_idx, ep_data in enumerate(episodes_data):
            experiences = ep_data.get("experiences", [])
            assigned = 0
            seg_start = 0
            current_tag = None
            for i, exp in enumerate(experiences):
                intent = exp.get("intentions", "")
                m = _TAG_RE.match(intent.strip())
                tag = m.group(1).upper() if m else "EXECUTE"
                if tag not in _SUBGOAL_TAG_SET:
                    tag = _TAG_ALIASES.get(tag, "EXECUTE")

                if tag != current_tag and current_tag is not None:
                    info = tag_to_info.get(current_tag, {})
                    if info:
                        seg_info = dict(info)
                        seg_info["segment_start"] = seg_start
                        seg_info["segment_end"] = i
                        for t in range(seg_start, i):
                            experiences[t]["skills"] = seg_info
                            assigned += 1
                    seg_start = i
                elif current_tag is None:
                    seg_start = i
                current_tag = tag

            if current_tag and current_tag in tag_to_info:
                info = tag_to_info[current_tag]
                seg_info = dict(info)
                seg_info["segment_start"] = seg_start
                seg_info["segment_end"] = len(experiences)
                for t in range(seg_start, len(experiences)):
                    if experiences[t].get("skills") is None:
                        experiences[t]["skills"] = seg_info
                        assigned += 1

            for exp in experiences:
                if exp.get("skills") is None:
                    exp["skills"] = None

            if verbose and assigned > 0:
                print(f"      Episode {ep_idx}: {assigned}/{len(experiences)} steps assigned to skills")
        return

    for ep_idx, ep_data in enumerate(episodes_data):
        traj_id = f"traj_{ep_idx}"
        traj_skills = step_to_skill.get(traj_id, {})
        experiences = ep_data.get("experiences", [])
        assigned = 0
        for i, exp in enumerate(experiences):
            skill_info = traj_skills.get(i)
            if skill_info:
                exp["skills"] = skill_info
                assigned += 1
            elif exp.get("skills") is None:
                exp["skills"] = None
        if verbose and assigned > 0:
            print(f"      Episode {ep_idx}: {assigned}/{len(experiences)} steps assigned to skills")


# ═══════════════════════════════════════════════════════════════════════
# Cross-game archetype aggregation
# ═══════════════════════════════════════════════════════════════════════

def _extract_dominant_tag(intentions: List[str]) -> str:
    tag_counts: Dict[str, int] = defaultdict(int)
    for intent in intentions:
        m = _TAG_RE.match(intent.strip())
        if m:
            tag = m.group(1).upper()
            if tag in _SUBGOAL_TAG_SET:
                tag_counts[tag] += 1
            elif tag in _TAG_ALIASES:
                tag_counts[_TAG_ALIASES[tag]] += 1
    if not tag_counts:
        return "EXECUTE"
    return max(tag_counts, key=tag_counts.get)


def aggregate_cross_game_archetypes(
    all_catalogs: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Aggregate per-game skills into cross-game archetypes keyed by SUBGOAL_TAG."""
    all_skills: List[Dict[str, Any]] = []
    for game, catalog in all_catalogs.items():
        for skill_entry in catalog.values():
            entry = dict(skill_entry)
            entry["_game"] = game
            all_skills.append(entry)

    if not all_skills:
        return {}

    for skill in all_skills:
        tag_hints: List[str] = []
        for text in [skill.get("name", ""), skill.get("description", ""), skill.get("summary", "")]:
            for tag in SUBGOAL_TAGS:
                if tag.lower() in text.lower():
                    tag_hints.append(f"[{tag}] {text[:50]}")
        if not tag_hints:
            tag_hints = [f"[EXECUTE] {skill.get('name', '')}"]
        skill["_dominant_tag"] = _extract_dominant_tag(tag_hints)

    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for skill in all_skills:
        by_tag[skill["_dominant_tag"]].append(skill)

    archetypes: List[Dict[str, Any]] = []

    for tag, skills in sorted(by_tag.items()):
        games_involved = sorted(set(s["_game"] for s in skills))

        instances = []
        for s in skills:
            instances.append({
                "game": s["_game"],
                "skill_id": s["skill_id"],
                "skill_name": s.get("name", s["skill_id"]),
                "summary": s.get("summary", ""),
                "description": s.get("description", ""),
                "eff_add": s.get("eff_add", []),
                "eff_del": s.get("eff_del", []),
                "eff_event": s.get("eff_event", []),
                "n_instances": s.get("n_instances", 0),
            })

        archetype_id = f"archetype_{tag.lower()}"
        instances_desc = "\n".join(
            f"  - {inst['game']}: {inst['skill_name']} — {inst['description'][:80]}"
            for inst in instances[:8]
        )

        prompt = (
            f"You are a game-AI transfer learning expert. Analyze these skills that "
            f"share the strategic pattern [{tag}] across different games:\n"
            f"{instances_desc}\n\n"
            f"Generate a game-agnostic archetype that captures the shared strategic "
            f"pattern. The archetype should be useful for transferring knowledge to "
            f"new games.\n\n"
            f"RULES:\n"
            f"- ARCHETYPE_NAME: 3-6 word abstract name (NOT just '{tag} Pattern').\n"
            f"- DESCRIPTION: 1-2 sentences on the abstract strategy, NOT game-specific.\n"
            f"- TRANSFER_SUMMARY: must include pattern=, trigger=, and strategy= fields.\n\n"
            f"Reply in this EXACT format:\n"
            f"ARCHETYPE_NAME: <name>\n"
            f"DESCRIPTION: <description>\n"
            f"TRANSFER_SUMMARY: archetype={tag.lower()} | pattern=<abstract pattern> | "
            f"trigger=<when to use> | strategy=<core strategy> | games={','.join(games_involved)}\n"
        )

        archetype_name = f"{tag.title()} Pattern"
        archetype_desc = f"Cross-game {tag} pattern observed in {', '.join(games_involved)}."
        transfer_summary = (
            f"archetype={tag.lower()} | games={','.join(games_involved)} | "
            f"n_skills={len(instances)} | n_games={len(games_involved)}"
        )

        result = _ask_llm(prompt, model=model, max_tokens=400, temperature=0.3,
                          _caller=f"aggregate_archetype:{archetype_id}")
        if result:
            for line in result.split("\n"):
                line = line.strip()
                if line.upper().startswith("ARCHETYPE_NAME:"):
                    parsed = line[len("ARCHETYPE_NAME:"):].strip().strip('"').strip("'")
                    if 3 <= len(parsed) <= 80:
                        archetype_name = parsed
                elif line.upper().startswith("DESCRIPTION:"):
                    parsed = line[len("DESCRIPTION:"):].strip().strip('"').strip("'")
                    if len(parsed) > 10:
                        archetype_desc = parsed[:250]
                elif line.upper().startswith("TRANSFER_SUMMARY:"):
                    parsed = line[len("TRANSFER_SUMMARY:"):].strip().strip('"').strip("'")
                    if len(parsed) > 10:
                        transfer_summary = parsed[:HARD_SUMMARY_CHAR_LIMIT]

        archetype = {
            "archetype_id": archetype_id,
            "tag": tag,
            "name": archetype_name,
            "description": archetype_desc,
            "transfer_summary": transfer_summary,
            "games": games_involved,
            "n_skills": len(instances),
            "n_games": len(games_involved),
            "instances": instances,
        }
        archetypes.append(archetype)

        if verbose:
            print(f"    [{tag}] {archetype_name} — {len(instances)} skills across {len(games_involved)} game(s)")

    archetypes_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "pipeline": "skill_agents_grpo",
        "n_archetypes": len(archetypes),
        "n_total_skills": len(all_skills),
        "n_games": len(all_catalogs),
        "games": sorted(all_catalogs.keys()),
        "archetypes": archetypes,
    }

    archetypes_path = output_dir / "skill_archetypes.json"
    with open(archetypes_path, "w", encoding="utf-8") as f:
        json.dump(archetypes_data, f, indent=2, ensure_ascii=False)

    rag_entries: List[Dict[str, Any]] = []
    for arch in archetypes:
        rag_entries.append({
            "id": arch["archetype_id"],
            "type": "archetype",
            "tag": arch["tag"],
            "name": arch["name"],
            "text": arch["transfer_summary"],
            "description": arch["description"],
            "games": arch["games"],
            "n_skills": arch["n_skills"],
        })
        for inst in arch["instances"]:
            rag_entries.append({
                "id": f"{arch['archetype_id']}_{inst['game']}_{inst['skill_id']}",
                "type": "skill_instance",
                "tag": arch["tag"],
                "archetype": arch["archetype_id"],
                "game": inst["game"],
                "name": inst["skill_name"],
                "text": inst["summary"],
                "description": inst["description"],
            })

    rag_index_path = output_dir / "skill_rag_index.json"
    with open(rag_index_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_entries": len(rag_entries),
            "entries": rag_entries,
        }, f, indent=2, ensure_ascii=False)

    print(f"    Archetypes     → {archetypes_path}")
    print(f"    RAG index      → {rag_index_path}")

    return archetypes_data


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-8B Skill Bank Agent (GRPO pipeline) — extract skills from GPT-5.4 rollouts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # All games (default input: labeling/output/gpt54)\n"
            "  python -m scripts.qwen3_skillbank_agent\n\n"
            "  # Specific games\n"
            "  python -m scripts.qwen3_skillbank_agent --games tetris super_mario\n\n"
            "  # Quick test\n"
            "  python -m scripts.qwen3_skillbank_agent --one_per_game -v\n\n"
            "  # Enable GRPO training with LoRA\n"
            "  python -m scripts.qwen3_skillbank_agent --use_grpo --local_model Qwen/Qwen3-8B --one_per_game -v\n\n"
            "  # Custom dirs\n"
            "  python -m scripts.qwen3_skillbank_agent --input_dir path/to/labeled --output_dir path/to/output\n"
        ),
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help=f"Directory with labeled game sub-folders (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--games", type=str, nargs="+", default=None,
        help="Only process these games",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"LLM model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Max episodes per game",
    )
    parser.add_argument(
        "--one_per_game", action="store_true",
        help="Process only the first episode per game",
    )
    parser.add_argument(
        "--resegment", action="store_true",
        help="Re-run pipeline against seeded bank (second pass, doubles LLM cost)",
    )
    parser.add_argument(
        "--skip_archetypes", action="store_true",
        help="Skip cross-game archetype aggregation",
    )
    parser.add_argument(
        "--save_annotated", action="store_true",
        help="Save annotated episodes (with skills field populated) to output dir",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Preview what would be processed without running extraction",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint (skip completed games/episodes)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-step details",
    )

    # GRPO / local-model options
    parser.add_argument(
        "--use_grpo", action="store_true",
        help="Enable GRPO training loop: sample G ranking sets per segment, "
             "select best-of-G, and train LoRA adapters after each game",
    )
    parser.add_argument(
        "--local_model", type=str, default=None,
        help="HuggingFace model id or local path for the base LLM "
             "(e.g. Qwen/Qwen3-8B). When set, uses this model for local "
             "inference and GRPO training",
    )
    parser.add_argument(
        "--adapter_dir", type=str, default=None,
        help="Directory containing LoRA adapters to load (and save after "
             "GRPO training). Defaults to <output_dir>/lora_adapters",
    )
    parser.add_argument(
        "--grpo_group_size", type=int, default=4,
        help="GRPO group size G (number of samples per call, default: 4)",
    )
    parser.add_argument(
        "--grpo_train_every", type=int, default=0,
        help="Run GRPO train_step every N episodes (0 = once per game, default: 0)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

    input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_INPUT_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    if args.one_per_game:
        args.max_episodes = 1

    # Local model / GRPO setup
    _local_llm = None
    _grpo_orch = None
    _adapter_dir = args.adapter_dir

    if args.local_model:
        if _adapter_dir is None:
            _adapter_dir = str(output_dir / "lora_adapters")
        _local_llm = setup_local_model(args.local_model, adapter_dir=_adapter_dir)
        if args.use_grpo:
            _grpo_orch = setup_grpo_orchestrator(
                _local_llm, group_size=args.grpo_group_size,
            )
    elif args.use_grpo:
        print("[ERROR] --use_grpo requires --local_model (e.g. Qwen/Qwen3-8B)")
        sys.exit(1)

    vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

    game_files = find_episode_files(input_dir, games=args.games)

    if not game_files:
        print(f"[ERROR] No episode files found under: {input_dir}")
        sys.exit(1)

    total_episodes = sum(len(v) for v in game_files.values())

    print("=" * 78)
    print("  Qwen3-8B Skill Bank Agent (GRPO pipeline)")
    print("=" * 78)
    print(f"  Pipeline:      skill_agents_grpo")
    print(f"  LLM backend:   {args.local_model or args.model}")
    print(f"  vLLM endpoint: {vllm_url}")
    print(f"  GRPO:          {'ON (G=' + str(args.grpo_group_size) + ')' if args.use_grpo else 'off'}")
    if _adapter_dir and args.use_grpo:
        print(f"  Adapter dir:   {_adapter_dir}")
    print(f"  Input:         {input_dir}")
    print(f"  Output:        {output_dir}")
    print(f"  Games:         {', '.join(sorted(game_files.keys()))}")
    print(f"  Episodes:      {total_episodes} total")
    per_game = args.max_episodes if args.max_episodes else "all"
    print(f"  Per game:      {per_game} episode(s)")
    print(f"  Re-segment:    {args.resegment}")
    print(f"  Archetypes:    {'SKIP' if args.skip_archetypes else 'yes'}")
    print(f"  Save annotated:{args.save_annotated}")
    print(f"  Dry run:       {args.dry_run}")
    print(f"  Resume:        {args.resume}")
    print(f"  I/O recording: enabled (stage_io_log.json per game)")
    print("=" * 78)

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for game, gfiles in sorted(game_files.items()):
            n = len(gfiles[:args.max_episodes]) if args.max_episodes else len(gfiles)
            sample = gfiles[0]
            ep = load_episode(sample)
            n_steps = len(ep.get("experiences", []))
            has_intentions = any(
                exp.get("intentions") for exp in ep.get("experiences", [])
            )
            print(f"  {game}: {n} episode(s), sample has {n_steps} steps, "
                  f"intentions={'yes' if has_intentions else 'NO'}")
        print("\nRe-run without --dry_run to execute.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = ExtractionCheckpoint(output_dir)
    if not args.resume:
        checkpoint.reset()

    overall_t0 = time.time()
    all_stats: List[Dict[str, Any]] = []
    all_catalogs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_io_logs: List[StageIOLog] = []

    for game, gfiles in sorted(game_files.items()):
        episode_files = gfiles[:args.max_episodes] if args.max_episodes else gfiles

        # Resume: skip fully completed games
        if args.resume and checkpoint.is_game_complete(game):
            print(f"\n  [RESUME] Skipping {game} (already complete)")
            cat_path = output_dir / game / "skill_catalog.json"
            if cat_path.exists():
                try:
                    cat_data = json.loads(cat_path.read_text(encoding="utf-8"))
                    all_catalogs[game] = {
                        s["skill_id"]: s for s in cat_data.get("skills", [])
                    }
                except Exception:
                    pass
            summary_path = output_dir / game / "extraction_summary.json"
            if summary_path.exists():
                try:
                    all_stats.append(json.loads(summary_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
            continue

        resume_from_episode = 0
        if args.resume:
            resume_from_episode = checkpoint.completed_episode_count(game)
            if resume_from_episode > 0:
                print(f"\n  [RESUME] {game}: resuming from episode {resume_from_episode}")

        print(f"\n{'━' * 78}")
        print(f"  GAME: {game} ({len(episode_files)} episodes) [Qwen3-8B + skill_agents_grpo]")
        print(f"{'━' * 78}")

        game_out_dir = output_dir / game
        game_out_dir.mkdir(parents=True, exist_ok=True)

        game_t0 = time.time()

        game_episodes_data: List[Dict[str, Any]] = []
        for fp in episode_files:
            try:
                ep_data = load_episode(fp)
                n_steps = len(ep_data.get("experiences", []))
                has_intent = any(
                    exp.get("intentions") for exp in ep_data.get("experiences", [])
                )
                if not has_intent:
                    print(f"    [WARN] {fp.name}: no intentions found — labeling may not have been run")
                game_episodes_data.append(ep_data)
                if args.verbose:
                    print(f"    Loaded {fp.name}: {n_steps} steps")
            except Exception as exc:
                print(f"    [ERROR] Failed to load {fp.name}: {exc}")

        if not game_episodes_data:
            print(f"    [SKIP] No valid episodes for {game}")
            continue

        print(f"    Loaded {len(game_episodes_data)} episode(s)")

        # Enable GRPO wrappers for this game
        if _grpo_orch is not None:
            from skill_agents_grpo.infer_segmentation.episode_adapter import (
                grpo_scorer_factory,
                grpo_decode_fn,
            )
            _grpo_orch.enable_wrappers(
                segment_scorer_factory=grpo_scorer_factory,
                segment_decode_fn=grpo_decode_fn,
            )
            print(f"    [GRPO] Wrappers enabled for {game}")

        # Skill extraction (GRPO pipeline)
        try:
            agent, skill_catalog, sub_episodes, io_log = extract_skills_for_game(
                game_episodes_data, game,
                output_dir=game_out_dir,
                model=args.model,
                verbose=args.verbose,
                resegment=args.resegment,
                checkpoint=checkpoint,
                resume_from_episode=resume_from_episode,
            )
            annotate_episodes_with_skills(
                game_episodes_data, agent, skill_catalog,
                verbose=args.verbose,
            )
            n_skills = len(skill_catalog)
            n_sub = len(sub_episodes)
            print(f"    Extracted {n_skills} skill(s), {n_sub} sub-episode(s)")
            all_catalogs[game] = skill_catalog
            all_io_logs.append(io_log)
        except Exception as exc:
            print(f"    [ERROR] Skill extraction failed: {exc}")
            if args.verbose:
                traceback.print_exc()
            skill_catalog = {}
            sub_episodes = []

        # Save annotated episodes
        if args.save_annotated:
            for ep_data in game_episodes_data:
                ep_id = ep_data.get("episode_id", "unknown")
                matching_name = None
                for fp in episode_files:
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            orig = json.load(f)
                        if orig.get("episode_id") == ep_id:
                            matching_name = fp.name
                            break
                    except Exception:
                        continue
                if matching_name is None:
                    matching_name = f"episode_{game_episodes_data.index(ep_data):03d}.json"
                out_path = game_out_dir / matching_name
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)
                    if args.verbose:
                        print(f"    Saved annotated → {out_path}")
                except Exception as exc:
                    print(f"    [ERROR] Failed to save annotated episode: {exc}")

        # GRPO: train step + disable wrappers + save adapters
        if _grpo_orch is not None:
            try:
                grpo_stats = _grpo_orch.train_step()
                _grpo_orch.disable_wrappers()
                if grpo_stats:
                    print(f"    [GRPO] Train step complete: {grpo_stats}")
                else:
                    print(f"    [GRPO] Train step: no samples in buffer")
                if _local_llm is not None and _adapter_dir:
                    save_lora_adapters(_local_llm, _adapter_dir)
                    print(f"    [GRPO] Adapters saved → {_adapter_dir}")
            except Exception as exc:
                print(f"    [GRPO] Training/save failed: {exc}")
                if args.verbose:
                    traceback.print_exc()
                try:
                    _grpo_orch.disable_wrappers()
                except Exception:
                    pass

        game_elapsed = time.time() - game_t0
        stat = {
            "game": game,
            "pipeline": "skill_agents_grpo",
            "episodes_processed": len(game_episodes_data),
            "skills_extracted": len(skill_catalog),
            "sub_episodes": len(sub_episodes),
            "elapsed_seconds": round(game_elapsed, 1),
            "grpo_enabled": args.use_grpo,
        }
        all_stats.append(stat)

        summary_path = game_out_dir / "extraction_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "game": game,
                "model": args.model,
                "pipeline": "skill_agents_grpo",
                "timestamp": datetime.now().isoformat(),
                "input_dir": str(input_dir / game),
                **stat,
            }, f, indent=2, ensure_ascii=False)

        if skill_catalog:
            catalog_path = game_out_dir / "skill_catalog.json"
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump({
                    "game": game,
                    "model": args.model,
                    "pipeline": "skill_agents_grpo",
                    "timestamp": datetime.now().isoformat(),
                    "n_skills": len(skill_catalog),
                    "skills": list(skill_catalog.values()),
                }, f, indent=2, ensure_ascii=False)
            print(f"    Skill catalog → {catalog_path}")

        checkpoint.mark_game_complete(game)
        print(f"    Done in {game_elapsed:.1f}s")

    # Cross-game archetype aggregation
    n_archetypes = 0
    if not args.skip_archetypes and len(all_catalogs) >= 1:
        print(f"\n{'━' * 78}")
        print(f"  Cross-Game Skill Archetype Aggregation (Qwen3-8B + skill_agents_grpo)")
        print(f"{'━' * 78}")
        print(f"  Aggregating skills from {len(all_catalogs)} game(s) ...")
        t0 = time.time()
        try:
            arch_data = aggregate_cross_game_archetypes(
                all_catalogs, output_dir,
                model=args.model, verbose=args.verbose,
            )
            n_archetypes = arch_data.get("n_archetypes", 0)
            elapsed = time.time() - t0
            print(f"    Generated {n_archetypes} archetype(s) in {elapsed:.1f}s")
        except Exception as exc:
            print(f"    [ERROR] Archetype aggregation failed: {exc}")
            if args.verbose:
                traceback.print_exc()

    overall_elapsed = time.time() - overall_t0

    # Final summary
    print(f"\n{'=' * 78}")
    print("  SKILL BANK EXTRACTION COMPLETE (Qwen3-8B + skill_agents_grpo)")
    print(f"{'=' * 78}")
    total_processed = sum(s["episodes_processed"] for s in all_stats)
    total_skills = sum(s["skills_extracted"] for s in all_stats)
    total_sub_eps = sum(s["sub_episodes"] for s in all_stats)
    print(f"  Pipeline:           skill_agents_grpo")
    print(f"  LLM backend:        {args.local_model or args.model}")
    print(f"  GRPO:               {'ON (G=' + str(args.grpo_group_size) + ')' if args.use_grpo else 'off'}")
    print(f"  Episodes processed: {total_processed}")
    print(f"  Skills extracted:   {total_skills}")
    print(f"  Sub-episodes:       {total_sub_eps}")
    print(f"  Archetypes:         {n_archetypes}")
    print(f"  Elapsed:            {overall_elapsed:.1f}s")
    print(f"  Output:             {output_dir}")

    master = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "pipeline": "skill_agents_grpo",
        "grpo_enabled": args.use_grpo,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_elapsed_seconds": round(overall_elapsed, 1),
        "total_episodes_processed": total_processed,
        "total_skills_extracted": total_skills,
        "total_sub_episodes": total_sub_eps,
        "total_archetypes": n_archetypes,
        "per_game": all_stats,
    }
    master_path = output_dir / "extraction_batch_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)
    print(f"  Summary:            {master_path}")

    if all_catalogs:
        combined_path = output_dir / "skill_catalog_all.json"
        combined = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "pipeline": "skill_agents_grpo",
            "total_skills": total_skills,
            "per_game": {g: list(cat.values()) for g, cat in all_catalogs.items()},
        }
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Full catalog:       {combined_path}")

    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
