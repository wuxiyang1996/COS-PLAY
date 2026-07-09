#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-8B Decision Agent with Dual LoRA GRPO + Skill Bank.

Uses Qwen3-8B (served via vLLM) with two GRPO-trained LoRA adapters:

  - **skill_selection LoRA**: state + intention + top-k candidates → chosen skill
  - **action_taking LoRA**:  state + actions + skill guidance → chosen action

Mirrors the cold-start labeling pipeline (label_episodes_with_skills.py)
for consistent I/O format between offline labeling and online play.

Pipeline per step:
  1. Generate ``summary_state`` (deterministic via ``build_rag_summary``)
  2. Generate ``summary`` (summary_state + delta-aware strategic note)
  3. Skill selection via **skill_selection LoRA** (top-k → LLM pick)
  4. Generate ``intention`` (skill-aware, delta/urgency-grounded, [TAG] phrase)
  5. Action selection via **action_taking LoRA** (state + skill + intention)
  6. Record GRPO I/O for both LoRAs (action_taking.jsonl + skill_selection.jsonl)

GRPO output per episode (in grpo_data/ under each game dir):
  - action_taking.jsonl    — state + available actions → chosen action
  - skill_selection.jsonl  — state + intention + candidates → chosen skill

Both follow a shared schema for a single GRPO trainer:
  { "type", "game", "episode", "step",
    "prompt", "completion", "reward",
    + type-specific metadata }

Usage (from Game-AI-Agent root):

    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
    export VLLM_BASE_URL="http://localhost:8000/v1"

    # Single game, 3 episodes (vLLM only, no LoRA)
    python -m scripts.qwen3_decision_agent --games twenty_forty_eight --episodes 3

    # With dual LoRA GRPO adapters
    python -m scripts.qwen3_decision_agent --use_lora \\
        --local_model Qwen/Qwen3-8B \\
        --adapter_dir adapters/decision_agent/ \\
        --games tetris --episodes 5 -v

    # Save trained adapters after run
    python -m scripts.qwen3_decision_agent --use_lora \\
        --save_adapters adapters/decision_agent/ \\
        --episodes 3

    # One episode per game (run each game once), on GPU 0, verbose
    python -m scripts.qwen3_decision_agent --one_per_game --gpu 0 -v

    # Without skill bank (baseline)
    python -m scripts.qwen3_decision_agent --no-bank --episodes 3

    # Custom skill bank path
    python -m scripts.qwen3_decision_agent --bank /path/to/bank --episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# Imports
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer

from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
from evaluate_gamingagent.gym_like import make_gaming_env, list_games
from evaluate_gamingagent.game_configs import GAME_CONFIGS

from decision_agents.agent_helper import (
    get_state_summary,
    infer_intention,
    strip_think_tags,
    select_skill_from_bank,
    build_rag_summary,
    extract_game_facts,
    compact_text_observation,
    HARD_SUMMARY_CHAR_LIMIT,
    SUBGOAL_TAGS,
)

try:
    from decision_agents.agent_helper import _get_protocol_for_skill
except ImportError:
    _get_protocol_for_skill = None
from decision_agents.dummy_agent import (
    extract_action,
    GAME_GAMINGAGENT,
)

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

from skill_agents.skill_bank.bank import SkillBankMVP

try:
    from skill_agents.query import SkillQueryEngine
except ImportError:
    SkillQueryEngine = None

try:
    from env_wrappers.tetris_macro_wrapper import TetrisMacroActionWrapper
except ImportError:
    TetrisMacroActionWrapper = None

try:
    from env_wrappers.sokoban_nl_wrapper import SokobanNLWrapper
except ImportError:
    SokobanNLWrapper = None

# RAG embedding for semantic action matching
try:
    from rag import get_text_embedder
    from rag.embedding.base import TextEmbedderBase
except ImportError:
    get_text_embedder = None
    TextEmbedderBase = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_BANK_DIR = CODEBASE_ROOT / "labeling" / "output" / "gpt54_skillbank"
DEFAULT_OUTPUT_DIR = CODEBASE_ROOT / "test_rollout" / "decision_agent"

LMGAME_BENCH_NAMES = {"twenty_forty_eight", "sokoban", "candy_crush", "tetris"}

INTENTION_WORD_BUDGET = 15
_SUBGOAL_TAG_SET = frozenset(SUBGOAL_TAGS)

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


# ---------------------------------------------------------------------------
# Tag normalization & state-delta helpers (ported from labeling pipeline)
# ---------------------------------------------------------------------------

def _normalize_intention(raw: str) -> str:
    """Ensure intention has a valid ``[TAG] phrase`` format."""
    raw = raw.split("\n")[0].strip().strip('"').strip("'")
    if not raw.startswith("["):
        return f"[EXECUTE] {raw}"
    m = _TAG_RE.match(raw)
    if not m:
        return f"[EXECUTE] {raw}"
    tag = m.group(1).upper()
    rest = raw[m.end():].strip()
    if tag not in _SUBGOAL_TAG_SET:
        tag = _TAG_ALIASES.get(tag, "EXECUTE")
    return f"[{tag}] {rest}" if rest else f"[{tag}]"


def _compute_state_delta(prev_ss: str, curr_ss: str) -> str:
    """Return a compact diff between two ``summary_state`` key=value strings."""
    if not prev_ss or not curr_ss:
        return ""

    def _parse(ss: str) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for seg in ss.split(" | "):
            if "=" in seg:
                k, v = seg.split("=", 1)
                d[k.strip()] = v.strip()
        return d

    skip = {"game", "step", "phase"}
    p, c = _parse(prev_ss), _parse(curr_ss)
    changes: List[str] = []
    for k, v in c.items():
        if k in skip:
            continue
        pv = p.get(k)
        if pv is not None and pv != v:
            changes.append(f"{k}:{pv}->{v}")
    return ", ".join(changes[:5])


def _detect_urgency(summary_state: str, game_name: str) -> str:
    """Return urgency warning when absolute state values are critical."""
    def _val(key: str) -> Optional[float]:
        for seg in summary_state.split(" | "):
            seg = seg.strip()
            if seg.startswith(f"{key}="):
                try:
                    return float(seg.split("=", 1)[1].split(",")[0])
                except (ValueError, IndexError):
                    return None
        return None

    gn = game_name.lower()
    warnings: List[str] = []
    if gn == "tetris":
        h = _val("holes")
        sh = _val("stack_h")
        if h is not None and h > 25:
            warnings.append("severe holes—prioritise CLEAR or SURVIVE")
        if sh is not None and sh > 14:
            warnings.append("stack near ceiling—SURVIVE")
    elif gn in ("2048", "twenty_forty_eight"):
        e = _val("empty")
        if e is not None and e < 3:
            warnings.append("board nearly full—must MERGE now")
    elif "candy" in gn:
        m = _val("moves")
        if m is not None and m < 5:
            warnings.append("very few moves left—maximise every action")
    elif "mario" in gn:
        t = _val("time")
        if t is not None and t < 50:
            warnings.append("time running out—NAVIGATE quickly")
    return "; ".join(warnings)


# ---------------------------------------------------------------------------
# Deterministic summary_state + LLM-enriched summary + skill-aware intention
# ---------------------------------------------------------------------------

def generate_summary_state(
    state: str,
    game_name: str = "",
    step_idx: int = -1,
    total_steps: int = -1,
    reward: float = 0.0,
) -> str:
    """Compact ``key=value`` state summary (deterministic, 0 LLM calls)."""
    return build_rag_summary(
        state, game_name,
        step_idx=step_idx,
        total_steps=total_steps,
        reward=reward,
    )


def generate_summary_prose(
    state: str,
    game_name: str = "",
    summary_state: str = "",
    prev_summary_state: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    """``summary_state | note=<strategic assessment>`` (1 cheap LLM call)."""
    if not summary_state:
        summary_state = generate_summary_state(state, game_name)

    compact = compact_text_observation(state, max_chars=200)
    state_text = compact if compact else state[:1000]
    game_label = game_name.replace("_", " ") if game_name else "game"
    delta = _compute_state_delta(prev_summary_state, summary_state)
    delta_line = f"Changed since last step: {delta}\n" if delta else ""

    prompt = (
        f"{game_label}: {state_text}\n"
        f"{delta_line}"
        f"Key strategic note about the current threat or opportunity "
        f"(max 10 words, be specific to what changed).\n"
        f"Note:"
    )

    if ask_model is not None:
        try:
            note = ask_model(prompt, model=model, temperature=0.2, max_tokens=25)
            if note and not note.startswith("Error"):
                note = strip_think_tags(note).strip()
                note = note.split("\n")[0].strip().strip('"').strip("'")[:80]
                return f"{summary_state} | note={note}"[:HARD_SUMMARY_CHAR_LIMIT]
        except Exception:
            pass
    return summary_state[:HARD_SUMMARY_CHAR_LIMIT]


def generate_skill_aware_intention(
    state: str,
    action: str,
    game_name: str = "",
    summary_state: str = "",
    prev_intention: str = "",
    prev_summary_state: str = "",
    skill_guidance: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Produce ``[TAG] subgoal`` — grounded in facts, delta, urgency, and active skill."""
    tags_str = "|".join(SUBGOAL_TAGS)
    facts_line = f"Facts: {summary_state}\n" if summary_state else ""
    delta = _compute_state_delta(prev_summary_state, summary_state)
    delta_line = f"Changed: {delta}\n" if delta else ""
    urgency = _detect_urgency(summary_state, game_name)
    urgency_line = f"URGENCY: {urgency}\n" if urgency else ""
    prev_line = f"Previous subgoal: {prev_intention}\n" if prev_intention else ""
    shift_hint = (
        "IMPORTANT: If the situation changed significantly or urgency is high, "
        "pick a NEW tag that matches the new priority.\n"
        if delta or urgency else ""
    )

    skill_context = ""
    if skill_guidance and skill_guidance.get("skill_id"):
        sk_name = skill_guidance.get("skill_name", skill_guidance["skill_id"])
        sk_hint = skill_guidance.get("execution_hint", "")
        skill_context = f"Active skill: {sk_name}"
        if sk_hint:
            skill_context += f" — {sk_hint[:100]}"
        skill_context += "\n"

    game_label = game_name.replace("_", " ") if game_name else "game"
    compact = compact_text_observation(state, max_chars=200)
    state_text = compact if compact else state[:800]

    prompt = (
        f"{game_label}. Action: {action}\n"
        f"State: {state_text}\n"
        f"{facts_line}"
        f"{delta_line}"
        f"{urgency_line}"
        f"{skill_context}"
        f"{prev_line}"
        f"{shift_hint}"
        f"What subgoal? Reply ONLY: [TAG] phrase "
        f"(max {INTENTION_WORD_BUDGET} words)\n"
        f"Tags: {tags_str}\n"
        f"Subgoal:"
    )

    if ask_model is not None:
        try:
            result = ask_model(prompt, model=model, temperature=0.2, max_tokens=40)
            if result and not result.startswith("Error"):
                result = strip_think_tags(result).strip()
                return _normalize_intention(result)[:150]
        except Exception:
            pass

    fallback = infer_intention(state, game=game_name, model=model)
    if fallback:
        return _normalize_intention(f"[EXECUTE] {fallback}")[:150]
    return f"[EXECUTE] {action}"


def _skill_guidance_to_label(guidance: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert skill guidance dict to compact label for storage."""
    if guidance is None or not guidance.get("skill_id"):
        return None
    label: Dict[str, Any] = {
        "skill_id": guidance["skill_id"],
        "skill_name": guidance.get("skill_name", ""),
    }
    if guidance.get("execution_hint"):
        label["execution_hint"] = guidance["execution_hint"][:200]
    if guidance.get("why_selected"):
        label["why_selected"] = guidance["why_selected"][:200]
    protocol = guidance.get("protocol", {})
    if protocol and isinstance(protocol, dict):
        compact_proto: Dict[str, Any] = {}
        if protocol.get("steps"):
            compact_proto["steps"] = protocol["steps"][:7]
        if protocol.get("preconditions"):
            compact_proto["preconditions"] = protocol["preconditions"][:3]
        if protocol.get("success_criteria"):
            compact_proto["success_criteria"] = protocol["success_criteria"][:3]
        if compact_proto:
            label["protocol"] = compact_proto
    if guidance.get("confidence") is not None:
        label["confidence"] = guidance["confidence"]
    return label


# ---------------------------------------------------------------------------
# Dual LoRA GRPO infrastructure (skill_selection + action_taking)
# ---------------------------------------------------------------------------

class DualLoRAManager:
    """Manages two LoRA adapters for the decision agent GRPO training.

    Adapter 1 — ``skill_selection``: state + intention + candidates → chosen skill
    Adapter 2 — ``action_taking``:  state + actions + skill guidance → chosen action

    When no local model is available, falls back to the default vLLM endpoint.
    """

    def __init__(self):
        self._llm = None
        self._grpo_orchestrator = None
        self._enabled = False

    def setup(
        self,
        model_name_or_path: str,
        adapter_dir: Optional[str] = None,
        grpo_group_size: int = 4,
    ) -> None:
        """Initialize the local model with optional LoRA adapters."""
        try:
            from skill_agents_grpo.lora.config import MultiLoraConfig
            from skill_agents_grpo.lora.model import MultiLoraSkillBankLLM
        except ImportError:
            print("[DualLoRA] skill_agents_grpo.lora not available, using vLLM only")
            return

        adapter_paths: Dict[str, str] = {}
        if adapter_dir:
            _ad = Path(adapter_dir)
            for name in ("skill_selection", "action_taking"):
                candidate = _ad / name
                if candidate.exists():
                    adapter_paths[name] = str(candidate)

        cfg = MultiLoraConfig(
            base_model_name_or_path=model_name_or_path,
            adapter_paths=adapter_paths,
            allow_fallback_to_base_model=True,
        )
        self._llm = MultiLoraSkillBankLLM(cfg)
        MultiLoraSkillBankLLM.set_shared_instance(self._llm)
        self._enabled = True

        print(f"  [DualLoRA] Model: {model_name_or_path}")
        if adapter_paths:
            print(f"  [DualLoRA] Adapters: {list(adapter_paths.keys())}")

        try:
            from skill_agents_grpo.grpo.orchestrator import GRPOOrchestrator
            from skill_agents_grpo.grpo.config import GRPOConfig, StageGRPOConfig

            grpo_cfg = GRPOConfig(stage_configs={
                "skill_selection": StageGRPOConfig(
                    group_size=grpo_group_size, kl_coeff=0.02, lr=3e-5,
                    epochs_per_batch=3, temperature=0.7,
                ),
                "action_taking": StageGRPOConfig(
                    group_size=grpo_group_size, kl_coeff=0.05, lr=5e-5,
                    epochs_per_batch=2, temperature=0.7,
                ),
            })
            self._grpo_orchestrator = GRPOOrchestrator(self._llm, grpo_cfg)
            print(f"  [DualLoRA] GRPO orchestrator (G={grpo_group_size})")
        except (ImportError, Exception) as exc:
            print(f"  [DualLoRA] GRPO not available: {exc}")

    def save_adapters(self, adapter_dir: str) -> None:
        """Save trained LoRA adapters to disk."""
        if self._llm is None or not self._enabled:
            return
        _ad = Path(adapter_dir)
        _ad.mkdir(parents=True, exist_ok=True)
        if not getattr(self._llm, "_is_peft_model", False):
            return
        for name in ("skill_selection", "action_taking"):
            adapter_name = f"lora_{name}"
            loaded = getattr(self._llm, "_loaded_adapters", {})
            if adapter_name not in loaded:
                continue
            save_path = _ad / name
            save_path.mkdir(parents=True, exist_ok=True)
            try:
                self._llm._model.set_adapter(adapter_name)
                self._llm._model.save_pretrained(str(save_path))
                print(f"    [DualLoRA] Saved adapter '{name}' → {save_path}")
            except Exception as exc:
                print(f"    [DualLoRA] Save failed '{name}': {exc}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def call_skill_selection(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 256,
    ) -> Optional[str]:
        """Call the skill_selection LoRA (or fallback to vLLM)."""
        if self._enabled and self._llm is not None:
            try:
                return self._llm.generate(
                    prompt, adapter_name="lora_skill_selection",
                    temperature=temperature, max_tokens=max_tokens,
                )
            except Exception:
                pass
        if ask_model is not None:
            return ask_model(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
        return None

    def call_action_taking(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """Call the action_taking LoRA (or fallback to vLLM)."""
        if self._enabled and self._llm is not None:
            try:
                return self._llm.generate(
                    prompt, adapter_name="lora_action_taking",
                    temperature=temperature, max_tokens=max_tokens,
                )
            except Exception:
                pass
        if ask_model is not None:
            return ask_model(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
        return None


_dual_lora = DualLoRAManager()


# ---------------------------------------------------------------------------
# GRPO I/O recording per step
# ---------------------------------------------------------------------------

class GRPOStepRecord:
    """Captures prompt/completion/reward for one GRPO training sample."""

    def __init__(self, record_type: str, game: str, episode_id: str, step: int):
        self.type = record_type
        self.game = game
        self.episode = episode_id
        self.step = step
        self.prompt: str = ""
        self.completion: str = ""
        self.reward: float = 0.0
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": self.type,
            "game": self.game,
            "episode": self.episode,
            "step": self.step,
            "prompt": self.prompt,
            "completion": self.completion,
            "reward": self.reward,
        }
        d.update(self.metadata)
        return d


# ---------------------------------------------------------------------------
# Skill bank loading
# ---------------------------------------------------------------------------

def load_skill_bank(
    bank_path: str,
    *,
    use_query_engine: bool = True,
) -> Tuple[Any, Any]:
    """Load a SkillBankMVP from a JSONL file or directory."""
    bp = Path(bank_path)
    if bp.is_dir():
        candidates = ["bank.jsonl", "skill_bank.jsonl"]
        jsonl = None
        for c in candidates:
            if (bp / c).exists():
                jsonl = bp / c
                break
        if jsonl is None:
            jsonls = sorted(bp.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
            jsonl = jsonls[0] if jsonls else None
        if jsonl is None:
            print(f"[load_skill_bank] WARNING: no .jsonl found in {bp}, using empty bank.")
            return None, None
        bp = jsonl

    bank = SkillBankMVP(path=str(bp))
    bank.load()
    print(f"[load_skill_bank] Loaded {len(bank)} skills from {bp}")

    engine = None
    if use_query_engine and SkillQueryEngine is not None and len(bank) > 0:
        try:
            engine = SkillQueryEngine(bank)
            print(f"[load_skill_bank] SkillQueryEngine initialised")
        except Exception as exc:
            print(f"[load_skill_bank] SkillQueryEngine init failed: {exc}")
    return bank, engine


def get_skill_guidance(
    skill_bank: Any,
    state_text: str,
    game_name: str = "",
    intention: str = "",
    structured_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Query the skill bank for guidance using expanded state context.

    When *structured_state* is provided it is passed through to
    ``select_skill_from_bank`` so the ``SkillQueryEngine.select()`` path
    can compute applicability scores (the full find-apply protocol).
    """
    if skill_bank is None:
        return None
    key_parts = []
    if game_name:
        key_parts.append(game_name)
    if intention:
        key_parts.append(intention)
    key_parts.append(state_text[:1500])
    key = " ".join(key_parts)

    state_for_scoring = None
    if structured_state and isinstance(structured_state, dict):
        state_for_scoring = {
            k: (float(v) if isinstance(v, (int, float, bool)) else 1.0)
            for k, v in structured_state.items()
            if v is not None and str(v).strip()
        }

    try:
        result = select_skill_from_bank(
            skill_bank, key,
            current_state=state_for_scoring,
            top_k=1,
        )
        if result and result.get("skill_id"):
            if not result.get("skill_name"):
                underlying = (
                    getattr(skill_bank, "_bank", None)
                    or getattr(skill_bank, "bank", None)
                    or skill_bank
                )
                if hasattr(underlying, "get_skill"):
                    skill_obj = underlying.get_skill(result["skill_id"])
                    if skill_obj:
                        result["skill_name"] = skill_obj.name or result["skill_id"]
                        if not result.get("execution_hint"):
                            result["execution_hint"] = skill_obj.strategic_description or ""
            return result
    except Exception:
        pass
    return None


def format_skill_guidance_for_prompt(
    guidance: Optional[Dict[str, Any]],
    protocol_step_idx: int = 0,
    progress_summary: str = "",
) -> str:
    """Format skill guidance as text to inject into LLM prompts.

    When *protocol_step_idx* > 0 the current step is highlighted with
    ``>>`` so the LLM knows where it is in the protocol sequence.

    When *progress_summary* is provided (from ``_SkillTracker``), it is
    included to give the model context on completed vs remaining steps.
    """
    if guidance is None or not guidance.get("skill_id"):
        return ""

    parts = [f"\n--- Active Skill: {guidance.get('skill_name', guidance['skill_id'])} ---"]
    if guidance.get("execution_hint"):
        parts.append(f"  Strategy: {guidance['execution_hint'][:200]}")

    if progress_summary:
        parts.append(f"  Progress: {progress_summary}")

    protocol = guidance.get("protocol", {})
    steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
    if steps:
        parts.append(f"  Plan ({len(steps)} steps):")
        for i, step in enumerate(steps[:7], 1):
            marker = ">>" if (i - 1) == protocol_step_idx else "  "
            parts.append(f"  {marker} {i}. {step}")

    preconditions = protocol.get("preconditions", []) if isinstance(protocol, dict) else []
    if preconditions:
        parts.append(f"  Preconditions: {'; '.join(preconditions[:3])}")

    success = protocol.get("success_criteria", []) if isinstance(protocol, dict) else []
    if success:
        parts.append(f"  Done when: {'; '.join(success[:2])}")

    abort = protocol.get("abort_criteria", []) if isinstance(protocol, dict) else []
    if abort:
        parts.append(f"  Abort if: {'; '.join(abort[:2])}")

    parts.append("--- end skill ---\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Top-k skill candidate retrieval + LLM-based skill selection
# ---------------------------------------------------------------------------

def get_top_k_skill_candidates(
    skill_bank: Any,
    state_text: str,
    game_name: str = "",
    intention: str = "",
    structured_state: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve *top_k* skill candidates from the skill bank.

    Returns a list of guidance dicts (same schema as ``get_skill_guidance``),
    sorted by confidence (highest first).  When fewer than *top_k* skills
    are available, returns whatever is available.
    """
    if skill_bank is None:
        return []

    key_parts = []
    if game_name:
        key_parts.append(game_name)
    if intention:
        key_parts.append(intention)
    key_parts.append(state_text[:1500])
    key = " ".join(key_parts)

    state_for_scoring = None
    if structured_state and isinstance(structured_state, dict):
        state_for_scoring = {
            k: (float(v) if isinstance(v, (int, float, bool)) else 1.0)
            for k, v in structured_state.items()
            if v is not None and str(v).strip()
        }

    from decision_agents.agent_helper import _get_protocol_for_skill

    candidates: List[Dict[str, Any]] = []

    # Preferred path: SkillQueryEngine.select() returns rich SkillSelectionResult list
    if hasattr(skill_bank, "select"):
        try:
            results = skill_bank.select(
                key,
                current_state=state_for_scoring,
                current_predicates=state_for_scoring,
                top_k=top_k,
            )
            for r in (results or []):
                d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
                sid = d.get("skill_id")
                if not sid:
                    continue
                protocol = _get_protocol_for_skill(skill_bank, sid)
                d["protocol"] = protocol or d.get("protocol", {})
                _enrich_candidate(skill_bank, d)
                candidates.append(d)
        except Exception:
            pass

    if candidates:
        return candidates

    # Fallback: build candidates from all active skills so that
    # skill_selection GRPO can fire even without SkillQueryEngine.
    try:
        _bank = getattr(skill_bank, "_bank", None) or getattr(skill_bank, "bank", None) or skill_bank
        if hasattr(_bank, "get_skills_for_decision_agent"):
            all_views = _bank.get_skills_for_decision_agent()
            if len(all_views) >= 2:
                for v in all_views[:top_k]:
                    d = dict(v)
                    sid = d.get("skill_id")
                    if sid:
                        protocol = _get_protocol_for_skill(skill_bank, sid)
                        d["protocol"] = protocol or d.get("protocol", {})
                        d.setdefault("confidence", 0.5)
                        d.setdefault("relevance", 0.5)
                        _enrich_candidate(skill_bank, d)
                        candidates.append(d)
                if len(candidates) >= 2:
                    return candidates
    except Exception:
        pass

    # Final fallback: single best skill
    try:
        single = get_skill_guidance(
            skill_bank, state_text, game_name, intention, structured_state,
        )
        if single and single.get("skill_id"):
            return [single]
    except Exception:
        pass

    return []


def _enrich_candidate(skill_bank: Any, d: Dict[str, Any]) -> None:
    """Fill in missing skill_name / execution_hint from the bank.

    When no strategic description is available (e.g. co-evolution skills
    that only have predicate-based contracts), generates a compact
    hint from the contract effects so the action-taking prompt has
    *some* context about what the skill does.
    """
    if d.get("skill_name") and d.get("execution_hint"):
        return
    sid = d.get("skill_id")
    if not sid:
        return
    underlying = (
        getattr(skill_bank, "_bank", None)
        or getattr(skill_bank, "bank", None)
        or skill_bank
    )
    if hasattr(underlying, "get_skill"):
        skill_obj = underlying.get_skill(sid)
        if skill_obj:
            if not d.get("skill_name"):
                d["skill_name"] = skill_obj.name or sid
            if not d.get("execution_hint"):
                desc = skill_obj.strategic_description or ""
                if not desc:
                    desc = _hint_from_contract(skill_obj)
                d["execution_hint"] = desc


def _hint_from_contract(skill_obj: Any) -> str:
    """Build a compact strategy hint from a skill's contract effects."""
    contract = getattr(skill_obj, "contract", None)
    if contract is None:
        return ""
    eff_add = getattr(contract, "eff_add", None) or []
    eff_del = getattr(contract, "eff_del", None) or []
    if not eff_add and not eff_del:
        return ""

    def _clean(pred: str) -> str:
        return pred.replace("event.", "").replace("world.", "").replace("_", " ")

    parts = []
    if eff_add:
        parts.append("causes: " + ", ".join(_clean(p) for p in sorted(eff_add)[:4]))
    if eff_del:
        parts.append("ends: " + ", ".join(_clean(p) for p in sorted(eff_del)[:4]))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Skill selection prompt + LLM call
# ---------------------------------------------------------------------------

SKILL_SELECTION_SYSTEM_PROMPT = (
    "You are an expert game strategist. "
    "Given the current game state and a set of candidate strategies, "
    "choose the ONE strategy most likely to make progress.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences why this strategy fits the current state>\n"
    "SKILL: <number>\n"
)

SKILL_SELECTION_USER_TEMPLATE = (
    "Game state:\n{state_summary}\n\n"
    "Current intention: {intention}\n\n"
    "Available strategies (pick ONE by number):\n{candidates_text}\n\n"
    "Choose the best strategy. Output REASONING then SKILL number."
)


def _format_candidates_for_selection(candidates: List[Dict[str, Any]]) -> str:
    """Format candidate skills as a numbered menu for the selection prompt."""
    lines: List[str] = []
    for i, c in enumerate(candidates, 1):
        name = c.get("skill_name") or c.get("skill_id", f"strategy_{i}")
        hint = c.get("execution_hint", "")
        protocol = c.get("protocol", {})
        steps = protocol.get("steps", []) if isinstance(protocol, dict) else []

        lines.append(f"  {i}. {name}")
        if hint:
            lines.append(f"     Strategy: {hint[:150]}")
        if steps:
            step_text = " -> ".join(steps[:4])
            if len(steps) > 4:
                step_text += " -> ..."
            lines.append(f"     Plan: {step_text}")

        confidence = c.get("confidence")
        if confidence is not None:
            lines.append(f"     Confidence: {confidence:.2f}")
    return "\n".join(lines)


def parse_skill_selection(
    reply: str,
    n_candidates: int,
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, Optional[str]]:
    """Parse the LLM skill selection response.

    Returns (chosen_index, reasoning).  *chosen_index* is 0-based.
    Falls back to the highest-confidence candidate (index 0) on parse
    failure, since candidates are pre-sorted by confidence.
    """
    if not reply:
        return 0, None

    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    reasoning = None
    reasoning_m = re.search(
        r"REASONING\s*:\s*(.+?)(?=\nSKILL|\Z)", cleaned, re.DOTALL | re.IGNORECASE,
    )
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    # Try SKILL: <number>
    skill_m = re.search(r"SKILL\s*:\s*(\d+)", cleaned, re.IGNORECASE)
    if skill_m:
        idx = int(skill_m.group(1)) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning

    # Fallback: any standalone digit in the last 100 chars
    tail = cleaned[-100:]
    nums = re.findall(r"\b(\d+)\b", tail)
    for n_str in reversed(nums):
        idx = int(n_str) - 1
        if 0 <= idx < n_candidates:
            return idx, reasoning

    # Fallback: match candidate name in response text
    if candidates:
        cleaned_lower = cleaned.lower()
        for i, c in enumerate(candidates):
            name = (c.get("skill_name") or "").lower()
            if name and len(name) >= 4 and name in cleaned_lower:
                return i, reasoning

    return 0, reasoning


def select_skill_via_llm(
    candidates: List[Dict[str, Any]],
    state_summary: str,
    intention: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> Tuple[int, Optional[str], Optional[str]]:
    """LLM call #1 (skill_selection LoRA): select the best skill.

    Returns (chosen_index, reasoning, raw_prompt) — the prompt is returned
    so callers can record it for GRPO training.  Falls back to index 0 if
    the LLM is unavailable.
    """
    if not candidates:
        return 0, None, None
    if len(candidates) == 1:
        return 0, "only one candidate available", None

    candidates_text = _format_candidates_for_selection(candidates)
    user_content = SKILL_SELECTION_USER_TEMPLATE.format(
        state_summary=state_summary[:3000],
        intention=intention[:500],
        candidates_text=candidates_text,
    )
    prompt = SKILL_SELECTION_SYSTEM_PROMPT + "\n" + user_content

    reply = _dual_lora.call_skill_selection(
        prompt, model=model, temperature=temperature, max_tokens=256,
    )
    if reply and not reply.startswith("Error"):
        idx, reasoning = parse_skill_selection(reply, len(candidates), candidates)
        return idx, reasoning, prompt

    return 0, None, prompt


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert game-playing agent. "
    "You receive a game state and must choose exactly one action by its NUMBER.\n\n"
    "Rules:\n"
    "- Study the state carefully before choosing.\n"
    "- Consider which action makes the most progress toward winning.\n"
    "- NEVER repeat the same action more than 2 times in a row — try something different.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Output format (strict):\n"
    "SUBGOAL: [TAG] <your immediate objective in \u226415 words>\n"
    "REASONING: <1-2 sentences>\n"
    "ACTION: <number>\n"
)

MAX_REPEAT_ACTIONS = 2


# ---------------------------------------------------------------------------
# RAG-based action embedding matcher
# ---------------------------------------------------------------------------

class ActionEmbeddingMatcher:
    """Lazily-initialized semantic action matcher using RAG text embeddings.

    Embeds valid actions once per action-set change, then scores the LLM's
    raw output against them via cosine similarity.  Falls back gracefully
    to None when the embedder is unavailable.
    """

    def __init__(self):
        self._embedder = None
        self._init_attempted = False
        self._cached_actions: Optional[List[str]] = None
        self._cached_embeddings = None  # np.ndarray or None

    def _ensure_embedder(self):
        if self._init_attempted:
            return
        self._init_attempted = True
        if get_text_embedder is None:
            return
        try:
            self._embedder = get_text_embedder()
            print("[ActionEmbeddingMatcher] TextEmbedder loaded")
        except Exception as exc:
            print(f"[ActionEmbeddingMatcher] TextEmbedder unavailable: {exc}")

    def _embed_actions(self, actions: List[str]):
        action_key = tuple(actions)
        if self._cached_actions == action_key:
            return
        self._cached_actions = action_key
        try:
            import numpy as np
            self._cached_embeddings = self._embedder.encode(
                actions, prompt_name="passage",
            )
        except Exception:
            self._cached_embeddings = None

    def match(
        self,
        raw: str,
        valid_actions: List[str],
        threshold: float = 0.35,
    ) -> Optional[str]:
        """Return the best semantic match or None if below threshold."""
        self._ensure_embedder()
        if self._embedder is None or not valid_actions or not raw:
            return None
        try:
            import numpy as np
            self._embed_actions(valid_actions)
            if self._cached_embeddings is None:
                return None
            q_emb = self._embedder.encode(raw, prompt_name="query")
            q_emb = np.atleast_2d(q_emb).astype(np.float32)
            scores = (q_emb @ self._cached_embeddings.T).squeeze(0)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score >= threshold:
                return valid_actions[best_idx]
        except Exception:
            pass
        return None


_action_matcher = ActionEmbeddingMatcher()


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _fuzzy_match_action(raw: str, valid_actions: List[str]) -> Optional[str]:
    """Match *raw* against *valid_actions* with multi-strategy fallback.

    1. Exact match (case-insensitive)
    2. Numbered selection (e.g. "3" -> third action)
    3. Substring containment
    4. Edit distance (for short actions) or token overlap (for long ones)
    """
    raw = raw.strip()
    if not raw or not valid_actions:
        return None

    raw_lower = raw.lower().rstrip(".").strip()
    lower_map = {a.lower(): a for a in valid_actions}

    if raw_lower in lower_map:
        return lower_map[raw_lower]

    num_m = re.match(r"^(\d+)\.?\s*$", raw_lower)
    if num_m:
        idx = int(num_m.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]

    num_m2 = re.search(r"(?:^|\s)(\d+)\s*[.:\-]", raw_lower)
    if num_m2:
        idx = int(num_m2.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]

    best_substr_match = None
    best_substr_len = 0
    for canon_lower, canon in lower_map.items():
        if len(canon_lower) < 3 and len(raw_lower) > 5:
            continue
        if canon_lower in raw_lower or raw_lower in canon_lower:
            if len(canon_lower) > best_substr_len:
                best_substr_len = len(canon_lower)
                best_substr_match = canon
    if best_substr_match:
        return best_substr_match

    raw_core = re.sub(r"[^a-z0-9_]", "", raw_lower)
    best_core_match = None
    best_core_len = 0
    for canon_lower, canon in lower_map.items():
        canon_core = re.sub(r"[^a-z0-9_]", "", canon_lower)
        if len(canon_core) < 3 and len(raw_core) > 5:
            continue
        if raw_core and canon_core and (raw_core in canon_core or canon_core in raw_core):
            if len(canon_core) > best_core_len:
                best_core_len = len(canon_core)
                best_core_match = canon
    if best_core_match:
        return best_core_match

    if len(valid_actions) <= 20 and all(len(a) < 30 for a in valid_actions):
        best_dist = float("inf")
        best_action = None
        for a in valid_actions:
            d = _edit_distance(raw_lower, a.lower())
            threshold = max(2, len(a) // 3)
            if d < best_dist and d <= threshold:
                best_dist = d
                best_action = a
        if best_action:
            return best_action

    raw_tokens = set(re.split(r"[\s,_\-()=]+", raw_lower))
    raw_tokens = {t for t in raw_tokens if len(t) >= 2}
    if raw_tokens:
        best_overlap = 0
        best_action = None
        for a in valid_actions:
            a_tokens = set(re.split(r"[\s,_\-()=]+", a.lower()))
            a_tokens = {t for t in a_tokens if len(t) >= 2}
            overlap = len(raw_tokens & a_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_action = a
        if best_action and best_overlap >= max(1, len(raw_tokens) // 3):
            return best_action

    # RAG embedding semantic match as final fallback
    rag_match = _action_matcher.match(raw, valid_actions)
    if rag_match:
        return rag_match

    return None


def parse_qwen_response(
    reply: str, valid_actions: List[str],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse SUBGOAL + REASONING + ACTION response (matches co-evolution format).

    Returns (action, reasoning, intention).
    """
    if not reply:
        return (valid_actions[0] if valid_actions else "stay"), None, None

    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    intention = None
    subgoal_m = re.search(
        r"SUBGOAL\s*:\s*(.+?)(?=\nREASONING|\nACTION|\Z)",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if subgoal_m:
        raw_sg = subgoal_m.group(1).strip().split("\n")[0].strip()
        intention = _normalize_intention(raw_sg)[:150] if raw_sg else None

    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    action_m = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if action_m:
        raw_action = action_m.group(1).strip()
        matched = _fuzzy_match_action(raw_action, valid_actions)
        if matched:
            return matched, reasoning, intention

    for pattern in [
        r"(?:choose|select|pick|answer)\s*(?:is\s*)?:?\s*(.+?)(?:\n|$)",
        r"(?:I (?:will|choose|select|pick))\s+(.+?)(?:\n|\.|$)",
    ]:
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            matched = _fuzzy_match_action(m.group(1).strip(), valid_actions)
            if matched:
                return matched, reasoning, intention

    extracted = extract_action(cleaned, GAME_GAMINGAGENT, "Valid actions: " + ", ".join(valid_actions))
    if extracted:
        return extracted, reasoning, intention

    for a in valid_actions:
        if len(a) >= 3 and a.lower() in cleaned.lower():
            return a, reasoning, intention

    action_m_text = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    raw_for_rag = action_m_text.group(1).strip() if action_m_text else cleaned[-200:]
    rag_match = _action_matcher.match(raw_for_rag, valid_actions)
    if rag_match:
        return rag_match, reasoning, intention

    return (valid_actions[0] if valid_actions else "stay"), reasoning, intention


def _format_numbered_actions(action_names: List[str]) -> str:
    """Format actions as a numbered list for the LLM prompt."""
    return "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_names))


def _build_recent_context(recent_actions: List[str], recent_rewards: List[float]) -> str:
    """Build a short context string about recent actions to prevent repetition."""
    if not recent_actions:
        return ""
    lines = ["Recent actions and rewards:"]
    for a, r in zip(recent_actions[-5:], recent_rewards[-5:]):
        lines.append(f"  {a} -> reward {r:.1f}")
    total = sum(recent_rewards[-5:])
    if total == 0 and len(recent_actions) >= 3:
        lines.append("WARNING: Recent actions got 0 reward. Try a DIFFERENT action!")
    lines.append("")
    return "\n".join(lines) + "\n"


def qwen3_action(
    state_nl: str,
    action_names: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    skill_guidance: Optional[Dict[str, Any]] = None,
    recent_actions: Optional[List[str]] = None,
    recent_rewards: Optional[List[float]] = None,
    protocol_step_idx: int = 0,
    summary_state: str = "",
    intention: str = "",
    progress_summary: str = "",
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """LLM call #2 (action_taking LoRA): choose an action.

    Returns (action, reasoning, raw_prompt, parsed_intention).
    The parsed_intention is the SUBGOAL extracted from the LLM response,
    which the caller can use as the new intention for the next step.
    """
    recent_context = _build_recent_context(
        recent_actions or [], recent_rewards or [],
    )

    state_text = summary_state if summary_state else state_nl[:4000]
    prev_line = f"Previous subgoal: {intention}\n" if intention else ""
    skill_context = ""
    if skill_guidance and skill_guidance.get("skill_id"):
        sk_name = skill_guidance.get("skill_name", skill_guidance["skill_id"])
        sk_hint = skill_guidance.get("execution_hint", "")
        skill_context = f"Active skill: {sk_name}"
        if sk_hint:
            skill_context += f" \u2014 {sk_hint[:100]}"
        skill_context += "\n"

    tags_str = "|".join(SUBGOAL_TAGS)
    user_content = (
        f"Game state:\n\n{state_text}\n\n"
        f"{prev_line}{skill_context}{recent_context}"
        f"Available actions (pick ONE by number):\n"
        f"{_format_numbered_actions(action_names)}\n\n"
        f"Subgoal tags: {tags_str}\n"
        f"First state your SUBGOAL, then choose the best action.\n"
        f"Output SUBGOAL, REASONING, then ACTION number."
    )

    skill_text = format_skill_guidance_for_prompt(
        skill_guidance, protocol_step_idx, progress_summary=progress_summary,
    )
    prompt = SYSTEM_PROMPT + skill_text + "\n" + user_content

    reply = _dual_lora.call_action_taking(
        prompt, model=model, temperature=temperature, max_tokens=512,
    )
    if reply and not reply.startswith("Error"):
        action, reasoning, parsed_intention = parse_qwen_response(reply, action_names)
        return action, reasoning, prompt, parsed_intention

    return (action_names[0] if action_names else "stay"), None, prompt, None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class _SkillTracker:
    """Protocol-aware skill lifecycle tracker.

    Implements the *find-apply* protocol:
    - **Find**: triggered on first step and whenever ``should_reselect()``
      returns True (duration exceeded, zero-reward stall, or abort criteria
      matched in the current state).
    - **Apply**: tracks progress through protocol steps, checks
      success/abort criteria each step, and exposes the current protocol
      step index for richer prompt injection.

    Uses predicate-based criteria (``predicate_success``, ``predicate_abort``)
    when available, falling back to legacy keyword matching on the free-text
    ``success_criteria`` / ``abort_criteria``.

    Step advancement uses ``step_checks`` when available: the step index
    advances only when the check condition is met in the current state.
    Falls back to time-based advancement (one step per timestep).
    """

    def __init__(self):
        self.active_skill_id: Optional[str] = None
        self.active_skill_name: str = ""
        self.steps_on_skill: int = 0
        self.reward_on_skill: float = 0.0
        self.max_skill_duration: int = 10
        self.skill_switches: int = 0

        self._protocol: Optional[Dict[str, Any]] = None
        self._protocol_step_idx: int = 0
        self._success_criteria: List[str] = []
        self._abort_criteria: List[str] = []
        self._predicate_success: List[str] = []
        self._predicate_abort: List[str] = []
        self._step_checks: List[str] = []
        self._reselect_reason: str = ""

    @property
    def reselect_reason(self) -> str:
        return self._reselect_reason

    @property
    def protocol_step_idx(self) -> int:
        return self._protocol_step_idx

    @property
    def total_protocol_steps(self) -> int:
        if self._protocol and isinstance(self._protocol, dict):
            return len(self._protocol.get("steps", []))
        return 0

    def _check_criteria(self, state_text: str, is_abort: bool) -> Optional[str]:
        """Check predicate or keyword criteria against state.

        Returns a reason string if triggered, None otherwise.
        """
        from decision_agents.protocol_utils import (
            parse_summary_state, check_any_predicate, keyword_match,
        )
        preds = self._predicate_abort if is_abort else self._predicate_success
        texts = self._abort_criteria if is_abort else self._success_criteria
        label = "abort" if is_abort else "success"

        if preds:
            state_dict = parse_summary_state(state_text)
            if check_any_predicate(preds, state_dict):
                return f"{label}:predicate"

        for crit in texts:
            if keyword_match(crit, state_text):
                return f"{label}:{crit[:40]}"

        return None

    def should_reselect(
        self,
        guidance: Optional[Dict[str, Any]],
        state_text: str = "",
    ) -> bool:
        """Decide whether to force a skill re-selection.

        Checks (in order): no skill, duration exceeded, zero-reward stall,
        abort criteria (predicate then keyword), success criteria
        (predicate then keyword).
        """
        self._reselect_reason = ""
        if guidance is None or not guidance.get("skill_id"):
            self._reselect_reason = "no_skill"
            return True
        new_id = guidance["skill_id"]
        if new_id != self.active_skill_id:
            return False

        if self.steps_on_skill >= self.max_skill_duration:
            self._reselect_reason = "duration_exceeded"
            return True

        if self.steps_on_skill >= 4 and self.reward_on_skill <= 0:
            self._reselect_reason = "zero_reward_stall"
            return True

        if state_text:
            abort_reason = self._check_criteria(state_text, is_abort=True)
            if abort_reason:
                self._reselect_reason = abort_reason
                return True

            if self.steps_on_skill >= 2:
                success_reason = self._check_criteria(state_text, is_abort=False)
                if success_reason:
                    self._reselect_reason = success_reason
                    return True

        return False

    def update(
        self,
        skill_id: Optional[str],
        skill_name: str,
        reward: float,
        state_text: str = "",
    ):
        if skill_id != self.active_skill_id:
            self.active_skill_id = skill_id
            self.active_skill_name = skill_name
            self.steps_on_skill = 1
            self.reward_on_skill = reward
            self.skill_switches += 1
            self._protocol_step_idx = 0
        else:
            self.steps_on_skill += 1
            self.reward_on_skill += reward
            n_steps = self.total_protocol_steps
            if n_steps > 0:
                from decision_agents.protocol_utils import (
                    compute_step_advancement, parse_summary_state,
                )
                state_dict = parse_summary_state(state_text)
                self._protocol_step_idx = compute_step_advancement(
                    self._protocol_step_idx,
                    self._step_checks,
                    state_dict,
                    n_steps,
                )

    def get_progress_summary(self, state_text: str = "") -> str:
        """Build a short progress summary for prompt injection."""
        if not self._protocol or not isinstance(self._protocol, dict):
            return ""
        steps = self._protocol.get("steps", [])
        if not steps:
            return ""
        from decision_agents.protocol_utils import (
            build_progress_summary, parse_summary_state,
        )
        state_dict = parse_summary_state(state_text)
        return build_progress_summary(
            steps, self._step_checks, self._protocol_step_idx, state_dict,
        )

    def set_protocol(self, protocol: Optional[Dict[str, Any]]):
        """Load protocol and extract criteria for lifecycle checks."""
        self._protocol = protocol
        self._protocol_step_idx = 0
        self._success_criteria = []
        self._abort_criteria = []
        self._predicate_success = []
        self._predicate_abort = []
        self._step_checks = []
        if protocol and isinstance(protocol, dict):
            dur = protocol.get("expected_duration", 0)
            if isinstance(dur, (int, float)) and dur > 0:
                self.max_skill_duration = max(int(dur) + 3, 5)
            else:
                self.max_skill_duration = 10
            self._success_criteria = protocol.get("success_criteria", []) or []
            self._abort_criteria = protocol.get("abort_criteria", []) or []
            self._predicate_success = protocol.get("predicate_success", []) or []
            self._predicate_abort = protocol.get("predicate_abort", []) or []
            self._step_checks = protocol.get("step_checks", []) or []
        else:
            self.max_skill_duration = 10


def _apply_anti_repetition(
    action: str,
    valid_actions: List[str],
    recent_actions: List[str],
    recent_rewards: List[float],
) -> str:
    """If the same action was repeated MAX_REPEAT_ACTIONS times with no reward,
    pick a different action from valid_actions."""
    if len(recent_actions) < MAX_REPEAT_ACTIONS:
        return action
    tail = recent_actions[-MAX_REPEAT_ACTIONS:]
    tail_rewards = recent_rewards[-MAX_REPEAT_ACTIONS:]
    if all(a == action for a in tail) and sum(tail_rewards) <= 0:
        alternatives = [a for a in valid_actions if a != action]
        if alternatives:
            import random
            chosen = random.choice(alternatives)
            return chosen
    return action


def run_episode(
    game: str,
    max_steps: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    verbose: bool = False,
    skill_bank: Any = None,
) -> Tuple[Episode, Dict[str, Any], List[Dict[str, Any]]]:
    """Run one episode with dual-LoRA GRPO decision agent + skill bank.

    Pipeline per step (mirrors cold-start labeling):
      1. Generate ``summary_state`` (deterministic via ``build_rag_summary``)
      2. Generate ``summary`` (summary_state + delta-aware strategic note)
      3. Skill selection via **skill_selection LoRA** (top-k → LLM pick)
      4. Generate ``intention`` (skill-aware, delta/urgency-grounded)
      5. Action selection via **action_taking LoRA** (state + skill + intention)
      6. Record GRPO I/O for both LoRAs

    Returns (episode, stats, grpo_records).
    """
    game_cfg = GAME_CONFIGS.get(game)
    task = game_cfg.description if game_cfg else f"Play {game}"

    base_env = make_gaming_env(game=game, max_steps=max_steps)

    if game == "sokoban" and SokobanNLWrapper is not None:
        env = SokobanNLWrapper(base_env, reflect_every=3)
    elif game == "tetris" and TetrisMacroActionWrapper is not None:
        # Match training: wrap NL env with placement-level macro actions.
        env = TetrisMacroActionWrapper(GamingAgentNLWrapper(base_env))
    else:
        env = GamingAgentNLWrapper(base_env)

    obs_nl, info = env.reset()
    action_names = info.get("action_names", [])
    structured_state = info.get("structured_state")

    import uuid as _uuid
    episode_id = f"{game}_{_uuid.uuid4().hex[:8]}"

    experiences: List[Experience] = []
    grpo_records: List[Dict[str, Any]] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    current_summary_state = ""
    current_summary = ""
    current_intention = ""
    prev_summary_state = ""
    prev_intention = ""

    recent_actions: List[str] = []
    recent_rewards: List[float] = []
    skill_tracker = _SkillTracker()
    last_guidance: Optional[Dict[str, Any]] = None
    last_candidates: List[Dict[str, Any]] = []
    last_chosen_idx: int = 0
    last_skill_reasoning: Optional[str] = None

    while step_count < max_steps:
        step_actions = action_names if action_names else ["stay"]

        # ── 1. summary_state (deterministic, 0 LLM calls) ──────────
        summary_state = generate_summary_state(
            obs_nl, game_name=game,
            step_idx=step_count, total_steps=max_steps,
            reward=total_reward,
        )
        current_summary_state = summary_state

        # ── 2. summary (summary_state + delta-aware note, 1 LLM call)
        summary = generate_summary_prose(
            obs_nl, game_name=game,
            summary_state=summary_state,
            prev_summary_state=prev_summary_state,
            model=model,
        )
        current_summary = summary

        # ── 3. Skill selection (skill_selection LoRA) ───────────────
        need_reselect = skill_tracker.should_reselect(
            last_guidance,
            state_text=summary_state or obs_nl,
        )
        skill_select_prompt: Optional[str] = None

        if need_reselect or last_guidance is None:
            if verbose and skill_tracker.reselect_reason:
                print(f"    [skill-reselect] reason={skill_tracker.reselect_reason}")

            facts = extract_game_facts(obs_nl, game)
            step_structured = {k: v for k, v in facts.items() if v}

            candidates = get_top_k_skill_candidates(
                skill_bank,
                summary_state or obs_nl,
                game_name=game,
                intention=current_intention,
                structured_state=step_structured if step_structured else structured_state,
                top_k=3,
            )

            if candidates:
                chosen_idx, skill_reasoning, skill_select_prompt = select_skill_via_llm(
                    candidates,
                    state_summary=summary_state or obs_nl,
                    intention=current_intention,
                    model=model,
                    temperature=temperature,
                )
                guidance = candidates[chosen_idx]
                if skill_reasoning:
                    guidance["why_selected"] = skill_reasoning
                last_candidates = candidates
                last_chosen_idx = chosen_idx
                last_skill_reasoning = skill_reasoning

                if verbose:
                    cand_names = [
                        c.get("skill_name") or c.get("skill_id", "?")
                        for c in candidates
                    ]
                    chosen_name = cand_names[chosen_idx] if chosen_idx < len(cand_names) else "?"
                    reason_disp = (
                        (skill_reasoning[:80] + "...") if skill_reasoning and len(skill_reasoning) > 80
                        else skill_reasoning
                    )
                    print(
                        f"    [skill-select] candidates: {cand_names}\n"
                        f"    [skill-select] chosen: {chosen_name} (reason: {reason_disp!r})"
                    )

                skill_tracker.set_protocol(guidance.get("protocol"))
            else:
                guidance = None
                last_candidates = []
                last_chosen_idx = 0
                last_skill_reasoning = None

            last_guidance = guidance
        else:
            guidance = last_guidance

        # ── 4. Intention (skill-aware, delta/urgency grounded) ──────
        intention = generate_skill_aware_intention(
            obs_nl,
            action=recent_actions[-1] if recent_actions else "start",
            game_name=game,
            summary_state=summary_state,
            prev_intention=prev_intention,
            prev_summary_state=prev_summary_state,
            skill_guidance=guidance,
            model=model,
        )
        current_intention = intention

        # ── 5. Action selection (action_taking LoRA) ────────────────
        action, reasoning, action_prompt, parsed_intention = qwen3_action(
            state_nl=obs_nl,
            action_names=step_actions,
            model=model,
            temperature=temperature,
            skill_guidance=guidance,
            recent_actions=recent_actions,
            recent_rewards=recent_rewards,
            protocol_step_idx=skill_tracker.protocol_step_idx,
            summary_state=summary_state,
            intention=current_intention,
            progress_summary=skill_tracker.get_progress_summary(summary_state),
        )
        if parsed_intention:
            current_intention = parsed_intention

        action = _apply_anti_repetition(action, step_actions, recent_actions, recent_rewards)

        try:
            next_obs_nl, reward, terminated, truncated, next_info = env.step(action)
        except Exception as e:
            print(f"    [ERROR at step {step_count}] env.step failed: {e}")
            break

        done = terminated or truncated
        total_reward += reward
        next_action_names = next_info.get("action_names", action_names)
        next_structured_state = next_info.get("structured_state")

        recent_actions.append(str(action))
        recent_rewards.append(float(reward))

        skill_id = guidance.get("skill_id") if guidance else None
        skill_name_val = guidance.get("skill_name", "") if guidance else ""
        skill_tracker.update(skill_id, skill_name_val, float(reward), state_text=summary_state)

        # ── 6. Record GRPO I/O ─────────────────────────────────────
        # Action-taking GRPO record
        if action_prompt:
            try:
                action_num = step_actions.index(action) + 1
            except ValueError:
                action_num = 1
            subgoal_line = f"SUBGOAL: {current_intention}\n" if current_intention else ""
            action_completion = f"{subgoal_line}REASONING: {reasoning or 'Expert play.'}\nACTION: {action_num}"
            act_rec = GRPOStepRecord("action_taking", game, episode_id, step_count)
            act_rec.prompt = action_prompt
            act_rec.completion = action_completion
            act_rec.reward = float(reward)
            act_rec.metadata = {
                "chosen_action": str(action),
                "available_actions": list(step_actions),
                "summary_state": summary_state,
                "intention": current_intention,
                "active_skill": skill_id,
            }
            grpo_records.append(act_rec.to_dict())

        # Skill-selection GRPO record
        if skill_select_prompt and last_candidates and len(last_candidates) >= 2:
            sk_completion = (
                f"REASONING: {last_skill_reasoning}\nSKILL: {last_chosen_idx + 1}"
                if last_skill_reasoning
                else f"SKILL: {last_chosen_idx + 1}"
            )
            sk_rec = GRPOStepRecord("skill_selection", game, episode_id, step_count)
            sk_rec.prompt = skill_select_prompt
            sk_rec.completion = sk_completion
            sk_rec.reward = float(reward)
            sk_rec.metadata = {
                "chosen_idx": last_chosen_idx,
                "skill_candidates": [c.get("skill_id") for c in last_candidates],
                "chosen_skill_id": last_candidates[last_chosen_idx].get("skill_id") if last_chosen_idx < len(last_candidates) else None,
                "summary_state": summary_state,
                "intention": current_intention,
            }
            grpo_records.append(sk_rec.to_dict())

        # ── Build Experience ────────────────────────────────────────
        exp = Experience(
            state=obs_nl,
            action=str(action),
            reward=float(reward),
            next_state=next_obs_nl,
            done=done,
            intentions=current_intention if current_intention else None,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.summary_state = summary_state
        exp.summary = current_summary
        exp.available_actions = list(step_actions) if step_actions else None
        exp.interface = {"env_name": "gamingagent", "game_name": game}

        skills_label = _skill_guidance_to_label(guidance)
        exp.sub_tasks = skills_label.get("skill_name", skills_label["skill_id"]) if skills_label else None

        if last_candidates:
            exp.skill_candidates = [c.get("skill_id") for c in last_candidates]
            exp.skill_chosen_idx = last_chosen_idx
            exp.skill_reasoning = last_skill_reasoning

        experiences.append(exp)

        if verbose:
            intent_short = (current_intention[:60] + "...") if len(current_intention) > 60 else current_intention
            reason_short = (reasoning[:60] + "...") if reasoning and len(reasoning) > 60 else reasoning
            ss_short = (summary_state[:60] + "...") if len(summary_state) > 60 else summary_state
            skill_disp = skill_tracker.active_skill_name or "none"
            proto_total = skill_tracker.total_protocol_steps
            if proto_total > 0:
                skill_info = (
                    f"{skill_disp} (step {skill_tracker.steps_on_skill}/{skill_tracker.max_skill_duration}, "
                    f"proto {skill_tracker.protocol_step_idx+1}/{proto_total})"
                )
            else:
                skill_info = f"{skill_disp} (step {skill_tracker.steps_on_skill}/{skill_tracker.max_skill_duration})"
            print(
                f"  step {step_count}: action={action}, reward={reward:.2f}, "
                f"cum={total_reward:.2f}\n"
                f"    summary_state: {ss_short}\n"
                f"    intention: {intent_short}\n"
                f"    skill:     {skill_info}\n"
                f"    reasoning: {reason_short}"
            )

        prev_summary_state = summary_state
        prev_intention = current_intention
        obs_nl = next_obs_nl
        action_names = next_action_names
        structured_state = next_structured_state
        step_count += 1

        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="gamingagent",
        game_name=game,
        episode_id=episode_id,
        metadata={
            "done": terminated or truncated,
            "steps": step_count,
            "total_reward": total_reward,
            "model": model,
            "agent_type": "qwen3_8b_dual_lora_grpo",
            "final_intention": current_intention,
            "final_summary_state": current_summary_state,
            "final_summary": current_summary,
            "skill_switches": skill_tracker.skill_switches,
            "unique_actions": len(set(recent_actions)),
            "grpo_records_count": len(grpo_records),
        },
    )
    episode.set_outcome()

    stats = {
        "game": game,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "qwen3_8b_dual_lora_grpo",
        "skill_switches": skill_tracker.skill_switches,
        "unique_actions": len(set(recent_actions)),
        "grpo_records": len(grpo_records),
    }
    return episode, stats, grpo_records


def _try_alternate_skill(
    skill_bank: Any,
    state_text: str,
    game_name: str,
    intention: str,
    exclude_id: str,
) -> Optional[Dict[str, Any]]:
    """Try to find an alternative skill excluding *exclude_id*."""
    if skill_bank is None:
        return None
    bank = getattr(skill_bank, "bank", skill_bank) if not hasattr(skill_bank, "skill_ids") else skill_bank
    try:
        ids = list(bank.skill_ids)
    except AttributeError:
        return None
    alt_ids = [sid for sid in ids if sid != exclude_id]
    if not alt_ids:
        return None
    import random
    chosen = random.choice(alt_ids)
    has_get_skill = hasattr(bank, "get_skill")
    from decision_agents.agent_helper import _get_protocol_for_skill
    protocol = _get_protocol_for_skill(bank, chosen)
    skill_obj = bank.get_skill(chosen) if has_get_skill else None
    skill_name = ""
    execution_hint = ""
    if skill_obj:
        skill_name = skill_obj.name or chosen
        execution_hint = skill_obj.strategic_description or ""
    return {
        "skill_id": chosen,
        "skill_name": skill_name,
        "why_selected": f"alternate skill (prev exhausted)",
        "protocol": protocol,
        "execution_hint": execution_hint,
    }


# ---------------------------------------------------------------------------
# GRPO training data export (two JSONL files per game)
# ---------------------------------------------------------------------------

def export_grpo_episode_data(
    grpo_records: List[Dict[str, Any]],
    grpo_dir: Path,
) -> Dict[str, int]:
    """Export GRPO training data for both LoRAs from a single episode.

    Writes two append-mode JSONL files under *grpo_dir*:

    ``action_taking.jsonl``
        One row per step. Contains the full action-selection prompt
        (state + available actions + active skill guidance + intention)
        and the chosen action. Reward is the step reward.

    ``skill_selection.jsonl``
        One row per step where >= 2 skill candidates were available.
        Contains the skill-selection prompt (state + intention +
        numbered candidate menu) and the chosen skill.
        Reward is the step reward.

    Returns ``{"action": n_action, "skill": n_skill}``.
    """
    if not grpo_records:
        return {"action": 0, "skill": 0}

    grpo_dir.mkdir(parents=True, exist_ok=True)
    action_path = grpo_dir / "action_taking.jsonl"
    skill_path = grpo_dir / "skill_selection.jsonl"

    n_action = 0
    n_skill = 0

    with open(action_path, "a", encoding="utf-8") as f_act, \
         open(skill_path, "a", encoding="utf-8") as f_sk:
        for rec in grpo_records:
            rec_type = rec.get("type", "")
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if rec_type == "action_taking":
                f_act.write(line)
                n_action += 1
            elif rec_type == "skill_selection":
                f_sk.write(line)
                n_skill += 1

    return {"action": n_action, "skill": n_skill}


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _sanitize_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_keys(v) for v in obj]
    return obj


def run_game_rollouts(
    game_name: str,
    args: argparse.Namespace,
    game_dir: Path,
    skill_bank: Any = None,
) -> Dict[str, Any]:
    """Run all episodes for one game and save outputs."""
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    game_cfg = GAME_CONFIGS.get(game_name)
    if args.max_steps:
        effective_max_steps = args.max_steps
    elif game_cfg:
        effective_max_steps = game_cfg.max_steps
    else:
        effective_max_steps = 50

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(args.episodes):
        print(f"\n  [{game_name}] Episode {ep_idx + 1}/{args.episodes}")

        try:
            episode, stats, grpo_records = run_episode(
                game=game_name,
                max_steps=effective_max_steps,
                model=args.model,
                temperature=args.temperature,
                verbose=args.verbose,
                skill_bank=skill_bank,
            )

            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

            episode_buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = _sanitize_keys(episode.to_dict())
            ep_data["metadata"] = stats
            ep_path = game_dir / f"episode_{ep_idx:03d}.json"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            record = _sanitize_keys(episode.to_dict())
            record["rollout_metadata"] = stats
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            # Export GRPO training data (action_taking + skill_selection)
            grpo_game_dir = game_dir / "grpo_data"
            grpo_counts = export_grpo_episode_data(grpo_records, grpo_game_dir)
            n_act = grpo_counts["action"]
            n_sk = grpo_counts["skill"]
            if n_act > 0 or n_sk > 0:
                print(f"    GRPO data: {n_act} action + {n_sk} skill samples → {grpo_game_dir}")

        except Exception as e:
            print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
            traceback.print_exc()
            all_stats.append({
                "game": game_name,
                "episode_index": ep_idx,
                "error": str(e),
                "steps": 0,
                "total_reward": 0.0,
            })

    elapsed = time.time() - t0

    if len(episode_buffer) == 0:
        if game_dir.exists():
            shutil.rmtree(game_dir)
        print(f"\n  [INVALID] {game_name}: 0 successful episodes, removed output dir")
        return {"game": game_name, "total_episodes": 0, "invalid": True}

    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))
    print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    summary: Dict[str, Any] = {
        "game": game_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "qwen3_8b_decision_with_skillbank",
        "total_episodes": len(all_stats),
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    valid_stats = [s for s in all_stats if "error" not in s]
    if valid_stats:
        rewards = [s["total_reward"] for s in valid_stats]
        steps = [s["steps"] for s in valid_stats]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["mean_steps"] = sum(steps) / len(steps)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-8B Decision Agent with Skill Bank guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="Games to play (default: all available LMGame-Bench)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per game (default: 3)")
    parser.add_argument("--one_per_game", action="store_true",
                        help="Run exactly 1 episode per game (overrides --episodes)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (default: per-game config)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3)")
    parser.add_argument("--bank", type=str, default=None,
                        help=f"Skill bank path (default: {DEFAULT_BANK_DIR})")
    parser.add_argument("--no-bank", action="store_true",
                        help="Run without skill bank (baseline)")
    parser.add_argument("--no-query-engine", action="store_true",
                        help="Disable SkillQueryEngine")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device(s) to use, e.g. '0' or '0,1' (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")

    # Dual LoRA / GRPO arguments
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable dual LoRA adapters (skill_selection + action_taking)")
    parser.add_argument("--local_model", type=str, default=None,
                        help="Local model path for LoRA (default: same as --model)")
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Directory with LoRA adapters (skill_selection/, action_taking/)")
    parser.add_argument("--save_adapters", type=str, default=None,
                        help="Save trained LoRA adapters to this directory after run")
    parser.add_argument("--grpo_group_size", type=int, default=4,
                        help="GRPO group size for LoRA training (default: 4)")

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.one_per_game:
        args.episodes = 1

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine available games
    try:
        all_games = list_games()
    except Exception:
        all_games = sorted(LMGAME_BENCH_NAMES)

    if args.games:
        requested_games = args.games
    else:
        requested_games = all_games

    available_games: List[str] = []
    for g in requested_games:
        cfg = GAME_CONFIGS.get(g)
        if cfg is None:
            print(f"[WARNING] Game '{g}' not found, skipping.")
            continue
        if not cfg.available:
            print(f"[WARNING] Game '{g}' not available, skipping.")
            continue
        available_games.append(g)

    if not available_games:
        print("[ERROR] No games available.")
        sys.exit(1)

    # Load skill bank
    skill_bank_obj = None
    bank_path_str = "none"

    if not args.no_bank:
        bank_base = Path(args.bank) if args.bank else DEFAULT_BANK_DIR

        per_game_banks: Dict[str, Any] = {}
        for game_name in available_games:
            game_bank_dir = bank_base / game_name
            if game_bank_dir.exists():
                bank, engine = load_skill_bank(
                    str(game_bank_dir),
                    use_query_engine=not args.no_query_engine,
                )
                per_game_banks[game_name] = engine if engine is not None else bank
            else:
                bank_obj, engine = load_skill_bank(
                    str(bank_base),
                    use_query_engine=not args.no_query_engine,
                )
                if bank_obj is not None:
                    per_game_banks[game_name] = engine if engine is not None else bank_obj
                    break

        if per_game_banks:
            bank_path_str = str(bank_base)
        else:
            print(f"[WARNING] No skill bank found at {bank_base}")

    # ---- Setup dual LoRA if requested ----
    if args.use_lora:
        lora_model = args.local_model or args.model
        _dual_lora.setup(
            model_name_or_path=lora_model,
            adapter_dir=args.adapter_dir,
            grpo_group_size=args.grpo_group_size,
        )

    vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

    gpu_info = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

    print("=" * 78)
    print("  Qwen3-8B Decision Agent with Skill Bank + Dual LoRA GRPO")
    print("=" * 78)
    print(f"  Model:       {args.model}")
    print(f"  vLLM:        {vllm_url}")
    print(f"  GPU:         {gpu_info}")
    print(f"  Skill Bank:  {bank_path_str}")
    print(f"  Dual LoRA:   {'enabled' if _dual_lora.enabled else 'disabled (vLLM only)'}")
    if args.adapter_dir:
        print(f"  Adapters:    {args.adapter_dir}")
    print(f"  Games:       {', '.join(available_games)}")
    print(f"  Episodes:    {args.episodes} per game")
    print(f"  Output:      {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for gi, game_name in enumerate(available_games, 1):
        print(f"\n{'━' * 78}")
        print(f"  GAME {gi}/{len(available_games)}: {game_name} ({args.episodes} episode(s))")
        print(f"{'━' * 78}")

        game_skill_bank = None
        if not args.no_bank:
            game_skill_bank = per_game_banks.get(game_name)
            if game_skill_bank is None:
                first_bank = next(iter(per_game_banks.values()), None) if per_game_banks else None
                game_skill_bank = first_bank
            bank_size = len(game_skill_bank) if game_skill_bank and hasattr(game_skill_bank, "__len__") else "?"
            print(f"  Skill bank: {bank_size} skills loaded")

        game_run_dir = output_dir / game_name / run_timestamp
        summary = run_game_rollouts(game_name, args, game_run_dir, skill_bank=game_skill_bank)
        game_summaries.append(summary)

        print(f"\n  ✓ {game_name} done — "
              f"{summary.get('total_episodes', 0)} ep, "
              f"mean_r={summary.get('mean_reward', 0):.2f}" if "mean_reward" in summary else
              f"\n  ✓ {game_name} done")

    overall_elapsed = time.time() - overall_t0

    # ---- Save LoRA adapters if requested ----
    if args.save_adapters and _dual_lora.enabled:
        _dual_lora.save_adapters(args.save_adapters)
        print(f"  LoRA adapters saved → {args.save_adapters}")

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "qwen3_8b_dual_lora_grpo",
        "skill_bank": bank_path_str,
        "dual_lora_enabled": _dual_lora.enabled,
        "episodes_per_game": args.episodes,
        "total_elapsed_seconds": overall_elapsed,
        "games": available_games,
        "results": game_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / f"eval_summary_{run_timestamp}.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  DECISION AGENT EVALUATION COMPLETE")
    print(f"{'=' * 78}")
    total_eps = sum(
        s.get("total_episodes", 0)
        for s in game_summaries
        if not s.get("invalid")
    )
    print(f"  Games:       {len(available_games)}")
    print(f"  Episodes:    {total_eps}")
    print(f"  Elapsed:     {overall_elapsed:.1f}s")
    print(f"  Output:      {output_dir}")
    print(f"  Summary:     {master_path}")

    successful = [s for s in game_summaries if not s.get("invalid") and "mean_reward" in s]
    if successful:
        avg_reward = sum(s["mean_reward"] for s in successful) / len(successful)
        avg_steps = sum(s["mean_steps"] for s in successful) / len(successful)
        print(f"  Avg reward:  {avg_reward:.2f}")
        print(f"  Avg steps:   {avg_steps:.1f}")

    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
