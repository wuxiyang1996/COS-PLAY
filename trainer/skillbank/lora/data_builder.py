"""
Dataset builders for function-specific LoRA training.

Each builder produces a list of ``{"prompt": ..., "completion": ...}``
dicts from existing training artifacts (decoded trajectories, contracts,
etc.).  These are converted into a HuggingFace ``Dataset`` for SFT.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from skill_agents.lora.skill_function import SkillFunction

logger = logging.getLogger(__name__)


# ── Prompt templates per function ────────────────────────────────────

BOUNDARY_PROMPT_TEMPLATE = """\
You are analyzing a game agent's trajectory to identify key state facts (predicates) at each timestep.

For each timestep below, extract the important discrete state facts as key-value pairs.
Focus on facts that, when they CHANGE between consecutive steps, would signal a meaningful transition:
- Location or area the agent is in
- Items held, inventory changes
- Game phase, menu/UI mode
- Objectives completed or active
- Interaction targets (NPCs, objects)
- Agent status (alive, health level, role)

Timesteps:
{states_block}

Return JSON array (length {num_states}):"""

SEGMENT_PROMPT_TEMPLATE = """\
You are an expert at recognizing skills in agent trajectories.

A trajectory segment spans timesteps {segment_start} to {segment_end} (length {length}).

Observations:
{observations}

Actions:
{actions}

Candidate skills: [{skills}]

Rank ALL candidate skills from best fit to worst fit for this segment.

Return ONLY a JSON object:
{{"ranking": ["best_skill", "second_best", ...], "reasoning": "brief explanation"}}"""

CONTRACT_PROMPT_TEMPLATE = """\
You are analyzing skill effects from game trajectory segments.

Skill: {skill_id}
Number of instances: {n_instances}

Representative segment observations:
{segment_observations}

State predicates at segment start: {predicates_start}
State predicates at segment end: {predicates_end}

Summarize the effects of this skill as a JSON object:
{{"eff_add": ["predicates that become true"], "eff_del": ["predicates that become false"], "description": "one-line description"}}"""

RETRIEVAL_PROMPT_TEMPLATE = """\
You are a skill retrieval assistant for a game-playing agent.

The agent's current goal: {query}
Current state: {current_state}

Available skills:
{skill_list}

Rewrite the query to better match relevant skills, then rank the top skills.

Return ONLY a JSON object:
{{"rewritten_query": "improved query", "ranking": ["best_skill", ...], "reasoning": "brief explanation"}}"""


# ── Dataset builders ─────────────────────────────────────────────────

def build_boundary_dataset(
    trajectories: Sequence[Any],
    output_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build prompt/completion pairs from trajectory predicate extractions.

    Each trajectory produces multiple training examples (one per chunk).

    Parameters
    ----------
    trajectories : list
        ``TrajectoryForEM`` objects with frames containing predicates.
    output_path : str, optional
        If given, write the dataset as JSONL.

    Returns
    -------
    list[dict]
        ``[{"prompt": ..., "completion": ...}, ...]``
    """
    examples: List[Dict[str, str]] = []

    for traj in trajectories:
        frames = getattr(traj, "frames", [])
        if not frames:
            continue
        chunk_size = 30
        for start in range(0, len(frames), chunk_size):
            chunk = frames[start:start + chunk_size]
            states_block = "\n".join(
                f"  t={start + i}: {getattr(f, 'state', str(f))}"
                for i, f in enumerate(chunk)
            )
            prompt = BOUNDARY_PROMPT_TEMPLATE.format(
                states_block=states_block,
                num_states=len(chunk),
            )
            # Ground-truth: the per-frame predicates from EM
            preds = [
                getattr(f, "predicates", {}) for f in chunk
            ]
            completion = json.dumps(preds, default=str)
            examples.append({"prompt": prompt, "completion": completion})

    if output_path:
        _write_jsonl(examples, output_path)
    logger.info("Boundary dataset: %d examples", len(examples))
    return examples


def build_segment_dataset(
    trajectories: Sequence[Any],
    decode_results: Sequence[Any],
    skill_names: List[str],
    output_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build prompt/completion pairs from decoded segmentations.

    Each decoded segment becomes one training example where the
    completion is the correct skill ranking.
    """
    examples: List[Dict[str, str]] = []
    traj_map = {getattr(t, "traj_id", i): t for i, t in enumerate(trajectories)}

    for dr in decode_results:
        traj = traj_map.get(dr.traj_id)
        if traj is None:
            continue
        frames = getattr(traj, "frames", [])
        for seg in dr.segments:
            t_s, t_e = seg.t_start, seg.t_end
            obs = [str(getattr(frames[t], "state", "")) for t in range(t_s, min(t_e + 1, len(frames)))]
            acts = [str(getattr(frames[t], "action", "")) for t in range(t_s, min(t_e + 1, len(frames)))]
            skills_str = ", ".join(f'"{s}"' for s in skill_names)

            prompt = SEGMENT_PROMPT_TEMPLATE.format(
                segment_start=t_s,
                segment_end=t_e,
                length=t_e - t_s + 1,
                observations=str(obs[:20]),
                actions=str(acts[:20]),
                skills=skills_str,
            )
            ranking = [seg.skill_label]
            if hasattr(seg, "runner_up") and seg.runner_up:
                ranking.append(seg.runner_up)
            completion = json.dumps({"ranking": ranking, "reasoning": "from EM decode"})
            examples.append({"prompt": prompt, "completion": completion})

    if output_path:
        _write_jsonl(examples, output_path)
    logger.info("Segment dataset: %d examples", len(examples))
    return examples


def build_contract_dataset(
    contracts: Dict[str, Any],
    trajectories: Sequence[Any],
    decode_results: Sequence[Any],
    output_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build prompt/completion pairs for contract summarization.

    Each skill with a learned contract becomes one training example.
    """
    examples: List[Dict[str, str]] = []
    traj_map = {getattr(t, "traj_id", i): t for i, t in enumerate(trajectories)}

    seg_by_skill: Dict[str, list] = {}
    for dr in decode_results:
        for seg in dr.segments:
            seg_by_skill.setdefault(seg.skill_label, []).append((dr.traj_id, seg))

    for skill_id, contract in contracts.items():
        segs = seg_by_skill.get(skill_id, [])[:3]
        seg_obs = []
        for traj_id, seg in segs:
            traj = traj_map.get(traj_id)
            if traj:
                frames = getattr(traj, "frames", [])
                obs = [str(getattr(frames[t], "state", ""))
                       for t in range(seg.t_start, min(seg.t_end + 1, len(frames)))]
                seg_obs.append(str(obs[:10]))

        eff_add = sorted(getattr(contract, "eff_add", set()))
        eff_del = sorted(getattr(contract, "eff_del", set()))

        prompt = CONTRACT_PROMPT_TEMPLATE.format(
            skill_id=skill_id,
            n_instances=getattr(contract, "n_instances", 0),
            segment_observations="; ".join(seg_obs) if seg_obs else "N/A",
            predicates_start="N/A",
            predicates_end="N/A",
        )
        completion = json.dumps({
            "eff_add": eff_add,
            "eff_del": eff_del,
            "description": f"Skill {skill_id}",
        })
        examples.append({"prompt": prompt, "completion": completion})

    if output_path:
        _write_jsonl(examples, output_path)
    logger.info("Contract dataset: %d examples", len(examples))
    return examples


def build_retrieval_dataset(
    query_log: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build prompt/completion pairs from historical retrieval queries.

    Parameters
    ----------
    query_log : list[dict]
        Each entry: ``{"query": str, "current_state": dict, "skill_list": str,
        "best_skill": str, "ranking": list}``.
    """
    examples: List[Dict[str, str]] = []

    for entry in query_log:
        prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
            query=entry.get("query", ""),
            current_state=json.dumps(entry.get("current_state", {}), default=str),
            skill_list=entry.get("skill_list", ""),
        )
        ranking = entry.get("ranking", [entry.get("best_skill", "")])
        completion = json.dumps({
            "rewritten_query": entry.get("query", ""),
            "ranking": ranking,
            "reasoning": "from query log",
        })
        examples.append({"prompt": prompt, "completion": completion})

    if output_path:
        _write_jsonl(examples, output_path)
    logger.info("Retrieval dataset: %d examples", len(examples))
    return examples


# ── Dispatch ─────────────────────────────────────────────────────────

BUILDERS = {
    SkillFunction.BOUNDARY: build_boundary_dataset,
    SkillFunction.SEGMENT: build_segment_dataset,
    SkillFunction.CONTRACT: build_contract_dataset,
    SkillFunction.RETRIEVAL: build_retrieval_dataset,
}


# ── Helpers ──────────────────────────────────────────────────────────

def _write_jsonl(examples: List[Dict[str, str]], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples to %s", len(examples), path)


def load_jsonl_dataset(path: str) -> List[Dict[str, str]]:
    """Load a JSONL dataset previously written by a builder."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples
