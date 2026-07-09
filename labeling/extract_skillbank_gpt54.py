#!/usr/bin/env python
"""
Extract skills from **already-labeled** GPT-5.4 rollouts and build a
persistent Skill Bank.

Reads episodes from ``labeling/output/gpt54/<game>/episode_*.json`` (or any
directory tree with the same layout) and runs the full SkillBankAgent pipeline:

  Stage 1+2 — Boundary proposal  + skill-sequence decoding
  Stage 3   — Effects-only contract learning / verify / refine
  Stage 4   — Bank maintenance: split, merge, refine
  Materialize — Promote ``__NEW__`` clusters to real skills
  Evaluation  — Skill quality assessment (coherence, granularity, …)

Then generates:
  - GPT-5.4 skill names, descriptions, and RAG summaries
  - Per-game ``skill_bank.jsonl`` and ``skill_catalog.json``
  - Cross-game archetype aggregation (``skill_archetypes.json``)
  - Combined ``skill_catalog_all.json``

Usage (from Game-AI-Agent root):

    export OPENROUTER_API_KEY="sk-or-..."
    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"

    # All games
    python labeling/extract_skillbank_gpt54.py

    # Specific games
    python labeling/extract_skillbank_gpt54.py --games tetris super_mario

    # Quick test: one episode per game
    python labeling/extract_skillbank_gpt54.py --one_per_game -v

    # Custom input/output dirs
    python labeling/extract_skillbank_gpt54.py \\
        --input_dir labeling/output/gpt54 \\
        --output_dir labeling/output/gpt54_skillbank

    # Re-segment against seeded bank
    python labeling/extract_skillbank_gpt54.py --resegment --one_per_game
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
# Project imports
# ---------------------------------------------------------------------------
from decision_agents.agent_helper import (
    strip_think_tags,
    build_rag_summary,
    HARD_SUMMARY_CHAR_LIMIT,
    SUBGOAL_TAGS,
)

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

try:
    import openai
    from api_keys import openai_api_key, open_router_api_key
except (ImportError, AttributeError):
    openai = None
    openai_api_key = None
    open_router_api_key = None

try:
    from API_func import OPENROUTER_BASE
except ImportError:
    OPENROUTER_BASE = "https://openrouter.ai/api/v1"

from skill_agents.pipeline import SkillBankAgent, PipelineConfig
from skill_agents.stage3_mvp.schemas import (
    ExecutionHint,
    SegmentRecord,
    SkillEffectsContract,
    SubEpisodeRef,
)
from data_structure.experience import Episode, Experience, SubTask_Experience

try:
    from skill_agents.infer_segmentation.episode_adapter import (
        _extract_predicates,
    )
except ImportError:
    _extract_predicates = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_GPT54 = "gpt-5.4"

DEFAULT_INPUT_DIR = SCRIPT_DIR / "output" / "gpt54"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "gpt54_skillbank"

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


# ═══════════════════════════════════════════════════════════════════════
# LLM helpers
# ═══════════════════════════════════════════════════════════════════════

def _ask_gpt54(
    prompt: str,
    *,
    system: str = "",
    model: str = MODEL_GPT54,
    temperature: float = 0.2,
    max_tokens: int = 150,
) -> Optional[str]:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    if ask_model is not None:
        result = ask_model(
            full_prompt, model=model,
            temperature=temperature, max_tokens=max_tokens,
        )
        if result and not result.startswith("Error"):
            return strip_think_tags(result).strip()
    return None


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
    """Convert a labeled episode dict to an Episode object for skill_agents."""
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
# Skill naming / description via GPT-5.4
# ═══════════════════════════════════════════════════════════════════════

def generate_skill_name(
    skill_id: str,
    contract: SkillEffectsContract,
    game_name: str,
    sample_intentions: List[str],
    model: str = MODEL_GPT54,
) -> Tuple[str, str]:
    """Ask GPT-5.4 to generate a short skill name and RAG summary."""
    eff_add_str = ", ".join(sorted(contract.eff_add)[:8]) if contract.eff_add else "none"
    eff_del_str = ", ".join(sorted(contract.eff_del)[:8]) if contract.eff_del else "none"
    eff_event_str = ", ".join(sorted(contract.eff_event)[:5]) if contract.eff_event else "none"
    intentions_str = " | ".join(sample_intentions[:5]) if sample_intentions else "n/a"

    prompt = (
        f"Game: {game_name}\n"
        f"Skill ID: {skill_id}\n"
        f"Effects added: {eff_add_str}\n"
        f"Effects removed: {eff_del_str}\n"
        f"Events: {eff_event_str}\n"
        f"Sample intentions from segments: {intentions_str}\n\n"
        f"Generate:\n"
        f"1. A short skill name (2-5 words, imperative verb phrase)\n"
        f"2. A compact RAG summary in key=value format for embedding retrieval\n\n"
        f"Reply in this exact format:\n"
        f"NAME: <skill name>\n"
        f"SUMMARY: game=<game> | skill=<name> | effects=<top effects> | context=<when to use>\n"
    )

    result = _ask_gpt54(prompt, model=model, max_tokens=120, temperature=0.3)

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

    return name, rag_summary


def generate_skill_protocol(
    skill_id: str,
    name: str,
    description: str,
    contract: SkillEffectsContract,
    game_name: str,
    sample_intentions: List[str],
    sample_states: List[str],
    model: str = MODEL_GPT54,
) -> Dict[str, Any]:
    """Ask GPT-5.4 to generate actionable protocol fields for a skill.

    Returns a dict with preconditions, steps, success_criteria, abort_criteria.
    """
    eff_add_str = ", ".join(sorted(contract.eff_add)[:6]) if contract.eff_add else "none"
    eff_del_str = ", ".join(sorted(contract.eff_del)[:6]) if contract.eff_del else "none"
    eff_event_str = ", ".join(sorted(contract.eff_event)[:5]) if contract.eff_event else "none"
    intentions_str = " | ".join(sample_intentions[:4]) if sample_intentions else "n/a"
    states_str = " // ".join(s[:100] for s in sample_states[:3]) if sample_states else "n/a"

    prompt = (
        f"Game: {game_name}\n"
        f"Skill: {name} ({skill_id})\n"
        f"Description: {description}\n"
        f"Effects added: {eff_add_str}\n"
        f"Effects removed: {eff_del_str}\n"
        f"Events: {eff_event_str}\n"
        f"Sample intentions: {intentions_str}\n"
        f"Sample states: {states_str}\n\n"
        f"Generate an actionable protocol for a decision agent to follow when executing this skill.\n"
        f"Reply in this EXACT format (one item per line within each section):\n"
        f"PRECONDITIONS:\n- <when this skill should be invoked>\n"
        f"STEPS:\n- <step 1>\n- <step 2>\n- ...\n"
        f"SUCCESS_CRITERIA:\n- <how to know the skill succeeded>\n"
        f"ABORT_CRITERIA:\n- <when to abandon this skill>\n"
    )

    result = _ask_gpt54(prompt, model=model, max_tokens=500, temperature=0.3)

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
        """Drop last item if it looks cut off mid-sentence (no sentence-ending punctuation)."""
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
        abort_criteria = ["No progress after expected duration"]

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
    model: str = MODEL_GPT54,
) -> str:
    """Ask GPT-5.4 to generate a 1–2 sentence description for the skill."""
    eff_str = ", ".join(sorted(contract.eff_add | contract.eff_del)[:10])
    states_str = " // ".join(s[:100] for s in sample_states[:3]) if sample_states else "n/a"

    prompt = (
        f"Game: {game_name}\nSkill: {name} ({skill_id})\n"
        f"Effects: {eff_str}\n"
        f"Sample states where skill was executed: {states_str}\n\n"
        f"Write 1-2 sentences describing what this skill does and when to use it. "
        f"Be concrete and specific to the game. Max 40 words.\nDescription:"
    )

    result = _ask_gpt54(prompt, model=model, max_tokens=120, temperature=0.3)
    if result:
        desc = result.split("\n")[0].strip().strip('"').strip("'")
        if len(desc) > 200:
            cut = desc[:200].rfind(".")
            desc = desc[:cut + 1] if cut > 80 else desc[:200]
        return desc
    return f"Skill '{name}' in {game_name}: applies {eff_str[:80]}."


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

    for se in all_sub_episodes:
        skill_id = se.sub_task
        if not skill_id:
            continue

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
        skill.sub_episodes = refs
        skill.n_instances = max(skill.n_instances, len(refs))
        agent.bank.add_or_update_skill(skill)
        linked += len(refs)

    if verbose:
        print(f"    Linked {linked} sub-episode ref(s) across {len(skill_refs)} skill(s)")
    return linked


def _populate_execution_hints(
    agent: SkillBankAgent,
    skill_catalog: Dict[str, Dict[str, Any]],
    model: str = MODEL_GPT54,
    verbose: bool = False,
) -> int:
    """Generate ExecutionHint for skills that lack one."""
    updated = 0
    bank = agent.bank

    for sid in list(bank.skill_ids):
        skill = bank.get_skill(sid)
        if skill is None or skill.retired:
            continue
        if skill.execution_hint is not None:
            continue

        cat_entry = skill_catalog.get(sid, {})
        name = skill.name or cat_entry.get("name", sid)
        desc = skill.strategic_description or cat_entry.get("description", "")
        tag = (cat_entry.get("tag", "") or "").upper()

        preconditions = skill.protocol.preconditions[:2] if skill.protocol.preconditions else []
        success_crit = skill.protocol.success_criteria[:2] if skill.protocol.success_criteria else []

        termination_cues = list(success_crit) if success_crit else []
        if not termination_cues and skill.contract and skill.contract.eff_add:
            termination_cues = [f"{lit} achieved" for lit in sorted(skill.contract.eff_add)[:2]]
        if not termination_cues and skill.contract and skill.contract.eff_event:
            termination_cues = [f"{ev} observed" for ev in sorted(skill.contract.eff_event)[:2]]
        if not termination_cues:
            termination_cues = [f"{name} objective met"]

        failure_modes = []
        if tag in ("SURVIVE", "DEFEND"):
            failure_modes.append("Board state deteriorates despite defensive moves")
        elif tag == "MERGE":
            failure_modes.append("No merge opportunities available on any legal move")
        elif tag in ("POSITION", "SETUP"):
            failure_modes.append("Structure broken — anchor tile dislodged or ordering disrupted")
        elif tag == "CLEAR":
            failure_modes.append("Clearing move creates worse congestion than before")
        if not failure_modes:
            failure_modes.append("No progress toward skill objective after several moves")

        n_refs = len(skill.sub_episodes) if skill.sub_episodes else 0

        hint = ExecutionHint(
            common_preconditions=preconditions,
            common_target_objects=[],
            state_transition_pattern=f"[{tag}] {desc[:80]}" if tag else desc[:80],
            termination_cues=termination_cues,
            common_failure_modes=failure_modes,
            execution_description=desc[:150],
            n_source_segments=n_refs,
        )

        skill.execution_hint = hint
        bank.add_or_update_skill(skill)
        updated += 1

    if verbose and updated > 0:
        print(f"    Generated {updated} execution hint(s)")
    return updated


def intention_based_segmentation(
    episodes_data: List[Dict[str, Any]],
    game_name: str,
    episodes: Optional[List[Episode]] = None,
    model: str = MODEL_GPT54,
    verbose: bool = False,
    outcome_length: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[SubTask_Experience]]:
    """Fallback segmentation using intention tag transitions.

    Groups consecutive steps with the same [TAG] into segments, then asks
    GPT-5.4 to generate skill names and summaries per segment group.
    """
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

    sub_episodes: List[SubTask_Experience] = []
    if episodes:
        sub_episodes = _build_sub_episodes_from_tags(
            tag_segments, episodes, game_name, outcome_length=outcome_length,
        )
        if verbose:
            print(f"    Built {len(sub_episodes)} SubTask_Experience objects from tag segments")

    return tag_segments, skill_catalog, sub_episodes


# ═══════════════════════════════════════════════════════════════════════
# Core: Skill extraction for one game
# ═══════════════════════════════════════════════════════════════════════

def populate_skill_protocols(
    agent: SkillBankAgent,
    skill_catalog: Dict[str, Dict[str, Any]],
    episodes_data: List[Dict[str, Any]],
    game_name: str,
    model: str = MODEL_GPT54,
    verbose: bool = False,
) -> int:
    """Fill empty protocols on skills in the bank using GPT-5.4.

    Iterates all skills, generates a protocol for any skill whose protocol
    has no steps, and updates the bank in-place.  Returns the count of
    protocols generated.
    """
    from skill_agents.stage3_mvp.schemas import Protocol

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
    model: str = MODEL_GPT54,
    verbose: bool = False,
    resegment: bool = False,
) -> Tuple[SkillBankAgent, Dict[str, Dict[str, Any]], List[SubTask_Experience]]:
    """Run the full SkillBankAgent pipeline on labeled episodes for one game.

    Stages:
      1+2 — Boundary proposal + segmentation
      3   — Contract learning / verify / refine
      4   — Bank maintenance: split / merge / refine
      Materialize — Promote __NEW__ clusters
      Evaluation  — Skill quality assessment

    Falls back to intention-based segmentation if the pipeline produces no
    skills (common with few episodes).

    Returns (agent, skill_catalog, sub_episodes).
    """
    bank_path = str(output_dir / "skill_bank.jsonl")

    config = PipelineConfig(
        bank_path=bank_path,
        env_name="llm",
        merge_radius=5,
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

    episodes: List[Episode] = []
    for ep_data in episodes_data:
        try:
            episodes.append(dict_to_episode(ep_data))
        except Exception as exc:
            print(f"    [WARN] Failed to convert episode: {exc}")

    if not episodes:
        print(f"    [WARN] No episodes to segment for {game_name}")
        return agent, {}, []

    # ── Stage 1+2: boundary proposal + segmentation ──
    all_sub_episodes: List[SubTask_Experience] = []
    print(f"    Segmenting {len(episodes)} episode(s) via SkillBankAgent ...")
    for i, ep in enumerate(episodes):
        try:
            result, ep_sub_episodes = agent.segment_episode(ep, env_name="llm")
            all_sub_episodes.extend(ep_sub_episodes)
            n_segs = len(result.segments) if hasattr(result, "segments") else 0
            if verbose:
                print(f"      Episode {i}: {len(ep.experiences)} steps → {n_segs} segments, {len(ep_sub_episodes)} sub-episodes")
        except Exception as exc:
            print(f"      [WARN] Episode {i} segmentation failed: {exc}")
            if verbose:
                traceback.print_exc()

    # ── Stage 3: contract learning ──
    if agent._all_segments:
        try:
            agent.run_contract_learning()
        except Exception as exc:
            if verbose:
                print(f"      [WARN] Stage 3 contract learning failed: {exc}")

    # ── Stage 4: bank maintenance (split / merge / refine) ──
    if agent._all_segments and len(agent.skill_ids) > 0:
        try:
            maint_result = agent.run_bank_maintenance()
            if verbose:
                n_s = len(maint_result.split_results) if hasattr(maint_result, "split_results") else 0
                n_m = len(maint_result.merge_results) if hasattr(maint_result, "merge_results") else 0
                n_r = len(maint_result.refine_results) if hasattr(maint_result, "refine_results") else 0
                print(f"      Stage 4 bank maintenance: {n_s} splits, {n_m} merges, {n_r} refines")
        except Exception as exc:
            if verbose:
                print(f"      [WARN] Stage 4 bank maintenance failed: {exc}")

    # ── Materialize NEW skills ──
    try:
        n_materialized = agent.materialize_new_skills()
        if verbose and n_materialized > 0:
            print(f"      Materialized {n_materialized} new skill(s)")
    except Exception as exc:
        if verbose:
            print(f"      [WARN] Materialize new skills failed: {exc}")

    # ── Skill evaluation ──
    if len(agent.skill_ids) > 0:
        try:
            eval_summary = agent.run_evaluation()
            if verbose:
                n_eval = len(eval_summary.skill_reports) if hasattr(eval_summary, "skill_reports") else 0
                print(f"      Evaluation: {n_eval} skill(s) evaluated")
        except Exception as exc:
            if verbose:
                print(f"      [WARN] Skill evaluation failed: {exc}")

    pipeline_skills = len(agent.skill_ids)
    if pipeline_skills > 0:
        if verbose:
            print(f"    SkillBankAgent extracted {pipeline_skills} skill(s)")
    else:
        print(f"    SkillBankAgent produced 0 skills — falling back to intention-based segmentation")

    # ── Fallback: intention-based segmentation ──
    use_intention_fallback = pipeline_skills == 0
    skill_catalog: Dict[str, Dict[str, Any]] = {}

    if use_intention_fallback:
        _tag_segments, skill_catalog, fallback_sub_episodes = intention_based_segmentation(
            episodes_data, game_name, episodes=episodes,
            model=model, verbose=verbose,
        )
        all_sub_episodes = fallback_sub_episodes

        for sid, entry in skill_catalog.items():
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
                tag = entry.get("tag", "")
                if tag:
                    skill.tags = [tag]
                    skill.expected_tag_pattern = [tag]
                agent.bank.add_or_update_skill(skill)
    else:
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

    # ── Optional re-segmentation pass against seeded bank ──
    if resegment and len(agent.skill_ids) > 0 and episodes:
        print(f"    Re-segmenting {len(episodes)} episode(s) against seeded bank ({len(agent.skill_ids)} skills) ...")
        reseg_agent = SkillBankAgent(config=config)
        reseg_agent.bank = agent.bank

        reseg_sub_episodes: List[SubTask_Experience] = []
        for i, ep in enumerate(episodes):
            try:
                result, ep_sub = reseg_agent.segment_episode(ep, env_name="llm")
                reseg_sub_episodes.extend(ep_sub)
                n_segs = len(result.segments) if hasattr(result, "segments") else 0
                if verbose:
                    print(f"      Re-seg episode {i}: {len(ep.experiences)} steps → {n_segs} segments, {len(ep_sub)} sub-episodes")
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
                    reseg_agent.run_bank_maintenance()
                except Exception as exc:
                    if verbose:
                        print(f"      [WARN] Re-seg Stage 4 failed: {exc}")
            try:
                reseg_agent.materialize_new_skills()
            except Exception:
                pass
            if len(reseg_agent.skill_ids) > 0:
                try:
                    reseg_agent.run_evaluation()
                except Exception:
                    pass

        if reseg_sub_episodes:
            all_sub_episodes = reseg_sub_episodes
            agent = reseg_agent
            print(f"    Re-segmentation produced {len(reseg_sub_episodes)} sub-episodes, {len(reseg_agent.skill_ids)} skills")

    # ── Generate protocols for skills with empty protocols ──
    if len(agent.skill_ids) > 0:
        print(f"    Generating protocols for skills with empty protocols ...")
        try:
            n_protos = populate_skill_protocols(
                agent, skill_catalog, episodes_data, game_name,
                model=model, verbose=verbose,
            )
            if n_protos > 0:
                print(f"    Generated {n_protos} protocol(s)")
            else:
                print(f"    All skills already have protocols")
        except Exception as exc:
            print(f"    [WARN] Protocol generation failed: {exc}")
            if verbose:
                traceback.print_exc()

    # ── Link sub-episodes to skills in the bank ──
    if all_sub_episodes and len(agent.skill_ids) > 0:
        try:
            n_linked = _link_sub_episodes_to_skills(agent, all_sub_episodes, verbose=verbose)
            if n_linked > 0:
                print(f"    Linked {n_linked} sub-episode ref(s) to skills")
        except Exception as exc:
            print(f"    [WARN] Sub-episode linking failed: {exc}")
            if verbose:
                traceback.print_exc()

    # ── Generate execution hints for skills ──
    if len(agent.skill_ids) > 0:
        try:
            n_hints = _populate_execution_hints(
                agent, skill_catalog, model=model, verbose=verbose,
            )
            if n_hints > 0:
                print(f"    Generated {n_hints} execution hint(s)")
        except Exception as exc:
            print(f"    [WARN] Execution hint generation failed: {exc}")
            if verbose:
                traceback.print_exc()

    # ── Persist skill bank ──
    try:
        agent.save()
    except Exception as exc:
        print(f"    [WARN] Failed to save skill bank: {exc}")

    # ── Persist SubTask_Experience objects ──
    if all_sub_episodes:
        sub_ep_path = output_dir / "sub_episodes.json"
        try:
            sub_ep_data = {
                "game": game_name,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "n_sub_episodes": len(all_sub_episodes),
                "sub_episodes": [se.to_dict() for se in all_sub_episodes],
            }
            with open(sub_ep_path, "w", encoding="utf-8") as f:
                json.dump(sub_ep_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"    Sub-episodes ({len(all_sub_episodes)}) → {sub_ep_path}")
        except Exception as exc:
            print(f"    [WARN] Failed to save sub_episodes: {exc}")

    return agent, skill_catalog, all_sub_episodes


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
    model: str = MODEL_GPT54,
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
            f"These game skills share the strategic pattern [{tag}]:\n"
            f"{instances_desc}\n\n"
            f"Generate:\n"
            f"1. ARCHETYPE_NAME: A short abstract name for this cross-game pattern (3-6 words)\n"
            f"2. DESCRIPTION: 1-2 sentences describing the abstract strategic pattern (game-agnostic)\n"
            f"3. TRANSFER_SUMMARY: Compact key=value RAG summary for cross-game retrieval\n\n"
            f"Reply in this exact format:\n"
            f"ARCHETYPE_NAME: <name>\n"
            f"DESCRIPTION: <description>\n"
            f"TRANSFER_SUMMARY: archetype={tag.lower()} | pattern=<pattern> | "
            f"trigger=<when to use> | strategy=<core strategy> | games=<game list>\n"
        )

        archetype_name = f"{tag.title()} Pattern"
        archetype_desc = f"Cross-game {tag} pattern observed in {', '.join(games_involved)}."
        transfer_summary = (
            f"archetype={tag.lower()} | games={','.join(games_involved)} | "
            f"n_skills={len(instances)} | n_games={len(games_involved)}"
        )

        result = _ask_gpt54(prompt, model=model, max_tokens=200, temperature=0.3)
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
        description="Extract skills from labeled GPT-5.4 rollouts and build a Skill Bank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # All games (default input: labeling/output/gpt54)\n"
            "  python labeling/extract_skillbank_gpt54.py\n\n"
            "  # Specific games\n"
            "  python labeling/extract_skillbank_gpt54.py --games tetris super_mario\n\n"
            "  # Quick test\n"
            "  python labeling/extract_skillbank_gpt54.py --one_per_game -v\n\n"
            "  # Custom dirs\n"
            "  python labeling/extract_skillbank_gpt54.py --input_dir path/to/labeled --output_dir path/to/output\n"
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
        "--model", type=str, default=MODEL_GPT54,
        help=f"LLM model for skill naming/description (default: {MODEL_GPT54})",
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
        "--verbose", "-v", action="store_true",
        help="Print per-step details",
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

    # Validate API key
    has_key = bool(
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or (open_router_api_key and str(open_router_api_key).strip())
    )
    if not has_key:
        print("[WARNING] No API key detected. LLM calls will fail.")
        print("  Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

    # Discover episode files
    game_files = find_episode_files(input_dir, games=args.games)

    if not game_files:
        print(f"[ERROR] No episode files found under: {input_dir}")
        sys.exit(1)

    total_episodes = sum(len(v) for v in game_files.values())

    print("=" * 78)
    print("  GPT-5.4 Skill Bank Extraction from Labeled Rollouts")
    print("=" * 78)
    print(f"  Model:         {args.model}")
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
            print(f"  {game}: {n} episode(s), sample has {n_steps} steps, intentions={'yes' if has_intentions else 'NO'}")
        print("\nRe-run without --dry_run to execute.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    all_stats: List[Dict[str, Any]] = []
    all_catalogs: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for game, gfiles in sorted(game_files.items()):
        episode_files = gfiles[:args.max_episodes] if args.max_episodes else gfiles
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game} ({len(episode_files)} episodes)")
        print(f"{'━' * 78}")

        game_out_dir = output_dir / game
        game_out_dir.mkdir(parents=True, exist_ok=True)

        game_t0 = time.time()

        # Load labeled episodes
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

        # ── Skill extraction ──
        try:
            agent, skill_catalog, sub_episodes = extract_skills_for_game(
                game_episodes_data, game,
                output_dir=game_out_dir,
                model=args.model,
                verbose=args.verbose,
                resegment=args.resegment,
            )
            annotate_episodes_with_skills(
                game_episodes_data, agent, skill_catalog,
                verbose=args.verbose,
            )
            n_skills = len(skill_catalog)
            n_sub = len(sub_episodes)
            print(f"    Extracted {n_skills} skill(s), {n_sub} sub-episode(s)")
            all_catalogs[game] = skill_catalog
        except Exception as exc:
            print(f"    [ERROR] Skill extraction failed: {exc}")
            if args.verbose:
                traceback.print_exc()
            skill_catalog = {}
            sub_episodes = []

        # ── Save annotated episodes ──
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

        game_elapsed = time.time() - game_t0
        stat = {
            "game": game,
            "episodes_processed": len(game_episodes_data),
            "skills_extracted": len(skill_catalog),
            "sub_episodes": len(sub_episodes),
            "elapsed_seconds": round(game_elapsed, 1),
        }
        all_stats.append(stat)

        # Per-game summary
        summary_path = game_out_dir / "extraction_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "game": game,
                "model": args.model,
                "timestamp": datetime.now().isoformat(),
                "input_dir": str(input_dir / game),
                **stat,
            }, f, indent=2, ensure_ascii=False)

        # Per-game skill catalog
        if skill_catalog:
            catalog_path = game_out_dir / "skill_catalog.json"
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump({
                    "game": game,
                    "model": args.model,
                    "timestamp": datetime.now().isoformat(),
                    "n_skills": len(skill_catalog),
                    "skills": list(skill_catalog.values()),
                }, f, indent=2, ensure_ascii=False)
            print(f"    Skill catalog → {catalog_path}")

        print(f"    Done in {game_elapsed:.1f}s")

    # ── Cross-game archetype aggregation ──
    n_archetypes = 0
    if not args.skip_archetypes and len(all_catalogs) >= 1:
        print(f"\n{'━' * 78}")
        print(f"  Cross-Game Skill Archetype Aggregation")
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

    # ── Final summary ──
    print(f"\n{'=' * 78}")
    print("  SKILL BANK EXTRACTION COMPLETE")
    print(f"{'=' * 78}")
    total_processed = sum(s["episodes_processed"] for s in all_stats)
    total_skills = sum(s["skills_extracted"] for s in all_stats)
    total_sub_eps = sum(s["sub_episodes"] for s in all_stats)
    print(f"  Episodes processed: {total_processed}")
    print(f"  Skills extracted:   {total_skills}")
    print(f"  Sub-episodes:       {total_sub_eps}")
    print(f"  Archetypes:         {n_archetypes}")
    print(f"  Elapsed:            {overall_elapsed:.1f}s")
    print(f"  Output:             {output_dir}")

    # Master summary
    master = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
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

    # Combined skill catalog across all games
    if all_catalogs:
        combined_path = output_dir / "skill_catalog_all.json"
        combined = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "total_skills": total_skills,
            "per_game": {g: list(cat.values()) for g, cat in all_catalogs.items()},
        }
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Full catalog:       {combined_path}")

    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
