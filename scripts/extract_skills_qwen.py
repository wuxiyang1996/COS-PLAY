#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract skills from labeled episode trajectories using Qwen3-8B via vLLM.

Uses the SkillBankAgent pipeline with the IntentionSignalExtractor (Strategy B)
so that Stage 1 boundary proposals come from [TAG] intention transitions —
no LLM calls for boundary detection.  The LLM (Qwen3-8B) is only used for:

  - Stage 2: preference-based skill ranking (LLM teacher)
  - Skill naming and description generation
  - Cross-game archetype aggregation (optional)

Reads labeled episodes that already have ``[TAG] phrase`` intentions
(e.g. from ``labeling/label_and_extract_skills_gpt54.py`` Phase 1 or
``scripts/run_qwen3_8b_eval.py --label``).

Output structure (scripts/output/qwen_skills/):
  <game_name>/episode_NNN.json      Episode with skills populated
  <game_name>/skill_bank.jsonl      Persistent skill bank (contracts)
  <game_name>/skill_catalog.json    RAG-friendly skill catalog (per-game)
  skill_catalog_all.json            Combined catalog across games

Requirements:
  - vLLM serving Qwen/Qwen3-8B  (set VLLM_BASE_URL, default localhost:8000/v1)
  - Labeled episodes with [TAG] intentions in the input directory

Usage (from Game-AI-Agent root):

    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
    export VLLM_BASE_URL="http://localhost:8000/v1"

    # Extract from GPT-5.4 labeled episodes (all games)
    python -m scripts.extract_skills_qwen

    # Specific game(s), verbose
    python -m scripts.extract_skills_qwen --games candy_crush tetris -v

    # One episode per game (quick test)
    python -m scripts.extract_skills_qwen --one_per_game -v

    # Custom input dir
    python -m scripts.extract_skills_qwen --input_dir labeling/output/gpt54_skills

    # Dry run (preview, no saves)
    python -m scripts.extract_skills_qwen --dry_run --one_per_game
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
from skill_agents.pipeline import SkillBankAgent, PipelineConfig
from skill_agents.stage3_mvp.schemas import SkillEffectsContract
from data_structure.experience import Episode, Experience

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

try:
    from decision_agents.agent_helper import SUBGOAL_TAGS, strip_think_tags
except ImportError:
    SUBGOAL_TAGS = (
        "SETUP", "CLEAR", "MERGE", "ATTACK", "DEFEND",
        "NAVIGATE", "POSITION", "COLLECT", "BUILD", "SURVIVE",
        "OPTIMIZE", "EXPLORE", "EXECUTE",
    )
    strip_think_tags = lambda t: t

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-8B"

DEFAULT_INPUT_DIRS: List[Path] = [
    CODEBASE_ROOT / "labeling" / "output" / "gpt54_skills",
]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "qwen_skills"

_SUBGOAL_TAG_SET = frozenset(SUBGOAL_TAGS)
_TAG_RE = re.compile(r"\[(\w+)\]\s*")

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


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _ask_llm(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 200,
) -> Optional[str]:
    """Call the LLM via ask_model and strip think tags."""
    if ask_model is None:
        return None
    result = ask_model(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    if result and not result.startswith("Error"):
        return strip_think_tags(result).strip()
    return None


def _has_intentions(episodes_data: List[Dict[str, Any]]) -> bool:
    """Check whether episodes carry [TAG] intention annotations."""
    for ep in episodes_data[:3]:
        for exp in ep.get("experiences", [])[:10]:
            intent = exp.get("intentions", "")
            if intent and _TAG_RE.match(intent.strip()):
                return True
    return False


def _normalize_tag(raw: str) -> str:
    """Extract canonical tag from an intention string."""
    m = _TAG_RE.match((raw or "").strip())
    if not m:
        return "EXECUTE"
    tag = m.group(1).upper()
    if tag in _SUBGOAL_TAG_SET:
        return tag
    return _TAG_ALIASES.get(tag, "EXECUTE")


# ═══════════════════════════════════════════════════════════════════════
# Episode conversion
# ═══════════════════════════════════════════════════════════════════════

def _dict_to_episode(episode_data: Dict[str, Any]) -> Episode:
    """Convert a labeled episode dict to an Episode object."""
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
# Skill naming (via Qwen)
# ═══════════════════════════════════════════════════════════════════════

def _generate_skill_name(
    skill_id: str,
    contract: SkillEffectsContract,
    game_name: str,
    sample_intentions: List[str],
    model: str = DEFAULT_MODEL,
) -> Tuple[str, str]:
    """Ask LLM to generate a short skill name and RAG summary."""
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
        f"Reply ONLY in this exact format (no extra text):\n"
        f"NAME: <skill name>\n"
        f"SUMMARY: game=<game> | skill=<name> | effects=<top effects> | context=<when to use>\n"
    )

    name = skill_id.replace("_", " ").title()
    rag_summary = f"game={game_name} | skill={skill_id} | eff_add={eff_add_str}"

    result = _ask_llm(prompt, model=model, max_tokens=120, temperature=0.3)
    if result:
        for line in result.split("\n"):
            line = line.strip()
            if line.upper().startswith("NAME:"):
                parsed = line[5:].strip().strip('"').strip("'")
                if 2 <= len(parsed) <= 60:
                    name = parsed
            elif line.upper().startswith("SUMMARY:"):
                parsed = line[8:].strip().strip('"').strip("'")
                if len(parsed) > 10:
                    rag_summary = parsed[:300]

    return name, rag_summary


def _generate_skill_description(
    skill_id: str,
    name: str,
    contract: SkillEffectsContract,
    game_name: str,
    sample_states: List[str],
    model: str = DEFAULT_MODEL,
) -> str:
    """Ask LLM to generate a 1-2 sentence description."""
    eff_str = ", ".join(sorted(contract.eff_add | contract.eff_del)[:10])
    states_str = " // ".join(s[:100] for s in sample_states[:3]) if sample_states else "n/a"

    prompt = (
        f"Game: {game_name}\nSkill: {name} ({skill_id})\n"
        f"Effects: {eff_str}\n"
        f"Sample states where skill was executed: {states_str}\n\n"
        f"Write 1-2 sentences describing what this skill does and when to use it. "
        f"Be concrete and specific to the game. Max 40 words.\n"
        f"Reply ONLY with the description, no prefix.\nDescription:"
    )

    result = _ask_llm(prompt, model=model, max_tokens=60, temperature=0.3)
    if result:
        desc = result.split("\n")[0].strip().strip('"').strip("'")
        return desc[:200]
    return f"Skill '{name}' in {game_name}: applies {eff_str[:80]}."


# ═══════════════════════════════════════════════════════════════════════
# Intention-based segmentation (fallback)
# ═══════════════════════════════════════════════════════════════════════

def _intention_based_segmentation(
    episodes_data: List[Dict[str, Any]],
    game_name: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Fallback: group consecutive steps with the same [TAG] into segments."""
    tag_segments: List[Dict[str, Any]] = []

    for ep_idx, ep_data in enumerate(episodes_data):
        exps = ep_data.get("experiences", [])
        if not exps:
            continue
        current_tag = None
        seg_start = 0
        for i, exp in enumerate(exps):
            tag = _normalize_tag(exp.get("intentions", ""))
            if tag != current_tag:
                if current_tag is not None and i > seg_start:
                    tag_segments.append({
                        "ep_idx": ep_idx, "tag": current_tag,
                        "start": seg_start, "end": i,
                        "intentions": [exps[t].get("intentions", "") for t in range(seg_start, min(i, len(exps)))],
                        "states": [str(exps[t].get("summary_state", ""))[:150] for t in range(seg_start, min(i, len(exps)))],
                    })
                current_tag = tag
                seg_start = i
        if current_tag is not None and len(exps) > seg_start:
            tag_segments.append({
                "ep_idx": ep_idx, "tag": current_tag,
                "start": seg_start, "end": len(exps),
                "intentions": [exps[t].get("intentions", "") for t in range(seg_start, len(exps))],
                "states": [str(exps[t].get("summary_state", ""))[:150] for t in range(seg_start, len(exps))],
            })

    if not tag_segments:
        return [], {}

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
        sample_intentions = []
        sample_states = []
        for s in segs[:5]:
            sample_intentions.extend(s["intentions"][:3])
            sample_states.extend(s["states"][:3])

        contract = SkillEffectsContract(
            skill_id=skill_id,
            eff_add={f"{tag.lower()}_completed"},
            eff_del=set(),
            eff_event={f"tag_{tag.lower()}"},
            n_instances=len(segs),
        )

        name, rag_summary = _generate_skill_name(skill_id, contract, game_name, sample_intentions, model=model)
        description = _generate_skill_description(skill_id, name, contract, game_name, sample_states, model=model)
        contract.name = name
        contract.description = description

        skill_catalog[skill_id] = {
            "skill_id": skill_id, "name": name,
            "summary": rag_summary, "description": description,
            "tag": tag,
            "eff_add": sorted(contract.eff_add),
            "eff_del": sorted(contract.eff_del),
            "eff_event": sorted(contract.eff_event),
            "n_instances": len(segs), "total_steps": total_steps, "version": 1,
        }
        if verbose:
            print(f"      [{tag}] {name} — {len(segs)} segment(s), {total_steps} steps")
        for s in segs:
            s["skill_id"] = skill_id
            s["skill_name"] = name
            s["skill_summary"] = rag_summary
            s["description"] = description

    return tag_segments, skill_catalog


# ═══════════════════════════════════════════════════════════════════════
# Main extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_skills_for_game(
    episodes_data: List[Dict[str, Any]],
    game_name: str,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> Tuple[SkillBankAgent, Dict[str, Dict[str, Any]]]:
    """Run skill extraction using the SkillBankAgent with IntentionSignalExtractor.

    For labeled episodes with [TAG] intentions, uses ``env_name="intention"``
    (Strategy B) so Stage 1 boundaries come from tag transitions — no LLM
    calls for boundary proposal.

    Falls back to intention-based segmentation if the pipeline produces no skills.
    """
    bank_path = str(output_dir / "skill_bank.jsonl")

    has_intentions = _has_intentions(episodes_data)
    env_name = "intention" if has_intentions else "llm"
    if verbose:
        print(f"    env_name={env_name} (intentions detected: {has_intentions})")
        print(f"    model={model} — concurrency limited to 1 for local inference")

    config = PipelineConfig(
        bank_path=bank_path,
        env_name=env_name,
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
        max_concurrent_llm_calls=1,
        report_dir=str(output_dir / "reports"),
    )

    agent = SkillBankAgent(config=config)

    episodes = []
    for ep_data in episodes_data:
        try:
            episodes.append(_dict_to_episode(ep_data))
        except Exception as exc:
            print(f"    [WARN] Failed to convert episode: {exc}")

    if not episodes:
        print(f"    [WARN] No episodes to segment for {game_name}")
        return agent, {}

    print(f"    Segmenting {len(episodes)} episode(s) via SkillBankAgent (env={env_name}) ...")
    for i, ep in enumerate(episodes):
        try:
            result, sub_episodes = agent.segment_episode(ep, env_name=env_name)
            n_segs = len(result.segments) if hasattr(result, "segments") else 0
            if verbose:
                print(f"      Episode {i}: {len(ep.experiences)} steps → {n_segs} segments")
        except Exception as exc:
            print(f"      [WARN] Episode {i} segmentation failed: {exc}")
            if verbose:
                traceback.print_exc()

    if agent._all_segments:
        try:
            agent.run_contract_learning()
        except Exception:
            pass

    try:
        agent.materialize_new_skills()
    except Exception:
        pass

    pipeline_skills = len(agent.skill_ids)
    if pipeline_skills > 0:
        if verbose:
            print(f"    SkillBankAgent extracted {pipeline_skills} skill(s)")
    else:
        print(f"    SkillBankAgent produced 0 skills — falling back to intention-based segmentation")

    use_intention_fallback = pipeline_skills == 0
    skill_catalog: Dict[str, Dict[str, Any]] = {}

    if use_intention_fallback:
        _, skill_catalog = _intention_based_segmentation(
            episodes_data, game_name, model=model, verbose=verbose,
        )
        for sid, entry in skill_catalog.items():
            contract = SkillEffectsContract(
                skill_id=sid, name=entry["name"], description=entry["description"],
                eff_add=set(entry.get("eff_add", [])),
                eff_del=set(entry.get("eff_del", [])),
                eff_event=set(entry.get("eff_event", [])),
                n_instances=entry.get("n_instances", 0),
            )
            agent.bank.add_or_update(contract)
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

            name, rag_summary = _generate_skill_name(sid, contract, game_name, sample_intentions, model=model)
            description = _generate_skill_description(sid, name, contract, game_name, sample_states, model=model)
            contract.name = name
            contract.description = description
            agent.bank.add_or_update(contract)

            skill_catalog[sid] = {
                "skill_id": sid, "name": name,
                "summary": rag_summary, "description": description,
                "eff_add": sorted(contract.eff_add) if contract.eff_add else [],
                "eff_del": sorted(contract.eff_del) if contract.eff_del else [],
                "eff_event": sorted(contract.eff_event) if contract.eff_event else [],
                "n_instances": contract.n_instances,
                "version": contract.version,
            }
            if verbose:
                print(f"      Skill {sid}: {name}")

    try:
        agent.save()
    except Exception as exc:
        print(f"    [WARN] Failed to save skill bank: {exc}")

    return agent, skill_catalog


def annotate_episodes_with_skills(
    episodes_data: List[Dict[str, Any]],
    agent: SkillBankAgent,
    skill_catalog: Dict[str, Dict[str, Any]],
    verbose: bool = False,
) -> None:
    """Populate the ``skills`` field on each experience."""
    has_intention_skills = any("tag" in v for v in skill_catalog.values())

    if has_intention_skills:
        tag_to_info: Dict[str, Dict[str, Any]] = {}
        for sid, entry in skill_catalog.items():
            tag = entry.get("tag")
            if tag:
                tag_to_info[tag] = {
                    "skill_id": sid, "skill_name": entry.get("name", sid),
                    "skill_summary": entry.get("summary", ""),
                    "description": entry.get("description", ""),
                    "eff_add": entry.get("eff_add", []),
                    "eff_del": entry.get("eff_del", []),
                    "eff_event": entry.get("eff_event", []),
                }
        for ep_idx, ep_data in enumerate(episodes_data):
            exps = ep_data.get("experiences", [])
            assigned = 0
            seg_start = 0
            current_tag = None
            for i, exp in enumerate(exps):
                tag = _normalize_tag(exp.get("intentions", ""))
                if tag != current_tag and current_tag is not None:
                    info = tag_to_info.get(current_tag, {})
                    if info:
                        seg_info = dict(info, segment_start=seg_start, segment_end=i)
                        for t in range(seg_start, i):
                            exps[t]["skills"] = seg_info
                            assigned += 1
                    seg_start = i
                elif current_tag is None:
                    seg_start = i
                current_tag = tag
            if current_tag and current_tag in tag_to_info:
                seg_info = dict(tag_to_info[current_tag], segment_start=seg_start, segment_end=len(exps))
                for t in range(seg_start, len(exps)):
                    if exps[t].get("skills") is None:
                        exps[t]["skills"] = seg_info
                        assigned += 1
            for exp in exps:
                if exp.get("skills") is None:
                    exp["skills"] = None
            if verbose and assigned > 0:
                print(f"      Episode {ep_idx}: {assigned}/{len(exps)} steps assigned")
        return

    step_to_skill: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for seg in agent.segments:
        if seg.skill_label in ("__NEW__", "NEW"):
            continue
        entry = skill_catalog.get(seg.skill_label, {})
        if not entry:
            continue
        skill_info = {
            "skill_id": seg.skill_label,
            "skill_name": entry.get("name", seg.skill_label),
            "skill_summary": entry.get("summary", ""),
            "description": entry.get("description", ""),
            "segment_start": seg.t_start, "segment_end": seg.t_end,
            "eff_add": entry.get("eff_add", []),
            "eff_del": entry.get("eff_del", []),
            "eff_event": entry.get("eff_event", []),
        }
        for t in range(seg.t_start, seg.t_end):
            step_to_skill[seg.traj_id][t] = skill_info

    for ep_idx, ep_data in enumerate(episodes_data):
        traj_id = f"traj_{ep_idx}"
        traj_skills = step_to_skill.get(traj_id, {})
        exps = ep_data.get("experiences", [])
        assigned = 0
        for i, exp in enumerate(exps):
            info = traj_skills.get(i)
            if info:
                exp["skills"] = info
                assigned += 1
            elif exp.get("skills") is None:
                exp["skills"] = None
        if verbose and assigned > 0:
            print(f"      Episode {ep_idx}: {assigned}/{len(exps)} steps assigned")


# ═══════════════════════════════════════════════════════════════════════
# File discovery
# ═══════════════════════════════════════════════════════════════════════

def find_episode_files(
    input_dirs: List[Path], games: Optional[List[str]] = None,
) -> List[Path]:
    files: List[Path] = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        for gd in sorted(input_dir.iterdir()):
            if not gd.is_dir():
                continue
            game_name = gd.name
            if games is not None and game_name not in games:
                continue
            for fp in sorted(gd.glob("episode_*.json")):
                if fp.name == "episode_buffer.json":
                    continue
                files.append(fp)
    return files


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract skills from labeled episodes using Qwen3-8B via vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, nargs="*", default=None,
                        help="Input dir(s) with game sub-folders of labeled episodes")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="Only process these games")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"vLLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes per game")
    parser.add_argument("--one_per_game", action="store_true",
                        help="Process only the first episode per game")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview without saving")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    input_dirs = [Path(d) for d in args.input_dir] if args.input_dir else DEFAULT_INPUT_DIRS
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    if args.one_per_game:
        args.max_episodes = 1

    files = find_episode_files(input_dirs, games=args.games)
    if not files:
        dirs_str = ", ".join(str(d) for d in input_dirs)
        print(f"[ERROR] No episode files found under: {dirs_str}")
        sys.exit(1)

    game_files: Dict[str, List[Path]] = {}
    for fp in files:
        game_files.setdefault(fp.parent.name, []).append(fp)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

    print("=" * 78)
    print("  Qwen3-8B Skill Extraction (SkillBankAgent + IntentionSignalExtractor)")
    print("=" * 78)
    print(f"  Model:         {args.model}")
    print(f"  vLLM endpoint: {vllm_url}")
    for d in input_dirs:
        print(f"  Input:         {d}")
    print(f"  Output:        {output_dir}")
    print(f"  Games:         {', '.join(sorted(game_files.keys()))}")
    print(f"  Episodes:      {sum(len(v) for v in game_files.values())} total")
    per_game = args.max_episodes if args.max_episodes else "all"
    print(f"  Per game:      {per_game} episode(s)")
    print(f"  Dry run:       {args.dry_run}")
    print("=" * 78)

    overall_t0 = time.time()
    all_stats: List[Dict[str, Any]] = []
    all_catalogs: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for game, gfiles in sorted(game_files.items()):
        episode_files = gfiles[:args.max_episodes] if args.max_episodes else gfiles
        print(f"\n{'━' * 78}")
        print(f"  GAME: {game} ({len(episode_files)} episodes)")
        print(f"{'━' * 78}")

        game_out_dir = output_dir / game
        if not args.dry_run:
            game_out_dir.mkdir(parents=True, exist_ok=True)

        game_t0 = time.time()
        game_episodes_data: List[Dict[str, Any]] = []

        for fp in episode_files:
            out_path = game_out_dir / fp.name
            if not args.overwrite and not args.dry_run and out_path.exists():
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        game_episodes_data.append(json.load(f))
                    print(f"  [SKIP] {fp.name} (already processed)")
                    continue
                except Exception:
                    pass

            print(f"  Loading {fp.name} ...")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    episode_data = json.load(f)
            except Exception as exc:
                print(f"    [ERROR] Failed to load: {exc}")
                continue

            n_steps = len(episode_data.get("experiences", []))
            print(f"    {n_steps} steps")
            game_episodes_data.append(episode_data)

            if args.dry_run:
                exps = episode_data.get("experiences", [])
                preview = exps[0] if exps else {}
                print(f"    --- Preview (step 0) ---")
                print(f"    intentions: {preview.get('intentions', '<none>')!r}")
                print(f"    summary_state: {str(preview.get('summary_state', '<none>'))[:80]}")
                break

        if args.dry_run:
            continue

        # Skill extraction
        skill_catalog: Dict[str, Dict[str, Any]] = {}
        if game_episodes_data:
            print(f"\n  Skill extraction for {game} ...")
            t0 = time.time()
            try:
                agent, skill_catalog = extract_skills_for_game(
                    game_episodes_data, game,
                    output_dir=game_out_dir,
                    model=args.model,
                    verbose=args.verbose,
                )
                annotate_episodes_with_skills(
                    game_episodes_data, agent, skill_catalog, verbose=args.verbose,
                )
                elapsed = time.time() - t0
                print(f"    Extracted {len(skill_catalog)} skill(s) in {elapsed:.1f}s")
                all_catalogs[game] = skill_catalog
            except Exception as exc:
                print(f"    [ERROR] Skill extraction failed: {exc}")
                if args.verbose:
                    traceback.print_exc()

        # Save
        game_saved = 0
        for ep_idx, ep_data in enumerate(game_episodes_data):
            fname = f"episode_{ep_idx:03d}.json"
            for fp in episode_files:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        orig = json.load(f)
                    if orig.get("episode_id") == ep_data.get("episode_id"):
                        fname = fp.name
                        break
                except Exception:
                    continue

            out_path = game_out_dir / fname
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)
                print(f"    Saved → {out_path}")
                game_saved += 1
            except Exception as exc:
                print(f"    [ERROR] Failed to save: {exc}")

        game_elapsed = time.time() - game_t0
        all_stats.append({
            "game": game, "episodes": game_saved,
            "skills": len(skill_catalog), "elapsed_s": game_elapsed,
        })

        if skill_catalog:
            catalog_path = game_out_dir / "skill_catalog.json"
            with open(catalog_path, "w", encoding="utf-8") as f:
                json.dump({
                    "game": game, "model": args.model,
                    "timestamp": datetime.now().isoformat(),
                    "n_skills": len(skill_catalog),
                    "skills": list(skill_catalog.values()),
                }, f, indent=2, ensure_ascii=False)
            print(f"    Skill catalog → {catalog_path}")

    overall_elapsed = time.time() - overall_t0

    print(f"\n{'=' * 78}")
    print("  SKILL EXTRACTION COMPLETE")
    print(f"{'=' * 78}")
    total_skills = sum(s["skills"] for s in all_stats)
    total_eps = sum(s["episodes"] for s in all_stats)
    print(f"  Episodes:    {total_eps}")
    print(f"  Skills:      {total_skills}")
    print(f"  Elapsed:     {overall_elapsed:.1f}s")
    print(f"  Output:      {output_dir}")

    if not args.dry_run and all_catalogs:
        combined_path = output_dir / "skill_catalog_all.json"
        combined = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "total_skills": total_skills,
            "per_game": {g: list(cat.values()) for g, cat in all_catalogs.items()},
        }
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Full catalog:  {combined_path}")

        summary_path = output_dir / "extraction_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "elapsed_s": overall_elapsed,
                "total_skills": total_skills,
                "per_game": all_stats,
            }, f, indent=2, ensure_ascii=False)
        print(f"  Summary:       {summary_path}")

    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
