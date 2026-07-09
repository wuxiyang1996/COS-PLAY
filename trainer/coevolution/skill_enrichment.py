"""Post-update skill enrichment for the co-evolution loop.

Ports key techniques from ``labeling/extract_skillbank_gpt54.py`` into the
online co-evolution pipeline.  These run after Stage 3+4, enriching skills
with protocols, execution hints, expected durations, and outcome tracking —
data that the decision agent consumes during rollouts.

Runs synchronously inside the bank-update executor (CPU-only, no LLM calls).
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ── Tag constants (mirrors decision_agents.agent_helper.SUBGOAL_TAGS) ──

_SUBGOAL_TAGS = (
    "SETUP", "CLEAR", "MERGE", "ATTACK", "DEFEND",
    "NAVIGATE", "POSITION", "COLLECT", "BUILD", "SURVIVE",
    "OPTIMIZE", "EXPLORE", "EXECUTE",
)
_SUBGOAL_TAG_SET = frozenset(_SUBGOAL_TAGS)

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

_TAG_RE = re.compile(r"\[(\w+)\]")


# ── Tag-specific protocol / hint templates ──
# Ported from extract_skillbank_gpt54.py generate_skill_protocol() and
# _populate_execution_hints().

_TAG_PROTOCOL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "SETUP": {
        "preconditions": ["Board/state allows preparatory placement"],
        "steps": [
            "Assess current layout for setup opportunities",
            "Place elements to prepare for future gains",
            "Verify arrangement is stable",
        ],
        "success_criteria": ["Target arrangement achieved"],
        "abort_criteria": ["Setup impossible given current constraints"],
        "failure_modes": ["Structure broken — anchor dislodged or ordering disrupted"],
    },
    "CLEAR": {
        "preconditions": ["Clearable groups or lines exist"],
        "steps": [
            "Identify best clearing opportunity",
            "Execute clearing move",
            "Assess board state after clear",
        ],
        "success_criteria": ["Target elements cleared"],
        "abort_criteria": ["No clearing moves available"],
        "failure_modes": ["Clearing move creates worse congestion than before"],
    },
    "MERGE": {
        "preconditions": ["Merge-eligible pairs or groups present"],
        "steps": [
            "Locate highest-value merge opportunity",
            "Execute merge sequence",
            "Reposition for next merge",
        ],
        "success_criteria": ["Merge completed, value increased"],
        "abort_criteria": ["No merge opportunities on any legal move"],
        "failure_modes": ["No merge opportunities available on any legal move"],
    },
    "ATTACK": {
        "preconditions": ["Offensive opportunity identified"],
        "steps": [
            "Evaluate target priority",
            "Execute attack action",
            "Confirm damage or progress",
        ],
        "success_criteria": ["Target defeated or objective advanced"],
        "abort_criteria": ["Health critical or target unreachable"],
        "failure_modes": ["Overcommitted to attack while defense deteriorated"],
    },
    "DEFEND": {
        "preconditions": ["Threat detected requiring defensive response"],
        "steps": [
            "Identify primary threat",
            "Take defensive position or action",
            "Hold until threat passes",
        ],
        "success_criteria": ["Threat neutralized, state stabilized"],
        "abort_criteria": ["Defense untenable, must change strategy"],
        "failure_modes": ["Board state deteriorates despite defensive moves"],
    },
    "NAVIGATE": {
        "preconditions": ["Movement toward target is possible"],
        "steps": [
            "Determine path to destination",
            "Move toward target avoiding hazards",
            "Confirm arrival or approach",
        ],
        "success_criteria": ["Reached target location"],
        "abort_criteria": ["Path blocked or environment changed"],
        "failure_modes": ["Stuck in loop or path is blocked"],
    },
    "POSITION": {
        "preconditions": ["Positioning adjustment needed"],
        "steps": [
            "Assess optimal target position",
            "Move elements into alignment",
            "Verify position is stable",
        ],
        "success_criteria": ["Elements in desired positions"],
        "abort_criteria": ["Repositioning would worsen state"],
        "failure_modes": ["Structure broken — anchor tile dislodged or ordering disrupted"],
    },
    "SURVIVE": {
        "preconditions": ["State is critical, survival priority"],
        "steps": [
            "Identify most dangerous constraint",
            "Take action to relieve pressure",
            "Stabilize to avoid game-over",
        ],
        "success_criteria": ["Danger reduced, stable state restored"],
        "abort_criteria": ["Recovery impossible"],
        "failure_modes": ["Board state deteriorates despite defensive moves"],
    },
    "OPTIMIZE": {
        "preconditions": ["Improvement opportunity exists in current layout"],
        "steps": [
            "Analyze current inefficiencies",
            "Make targeted improvement move",
            "Verify improvement achieved",
        ],
        "success_criteria": ["Measurable state improvement"],
        "abort_criteria": ["Optimization would sacrifice critical position"],
        "failure_modes": ["Optimization broke a more important structure"],
    },
    "EXPLORE": {
        "preconditions": ["Unknown territory or options available"],
        "steps": [
            "Choose unexplored direction or option",
            "Investigate and gather information",
            "Update strategy based on findings",
        ],
        "success_criteria": ["New information or area discovered"],
        "abort_criteria": ["Exploration too risky given current state"],
        "failure_modes": ["Exploration consumed resources with no useful discovery"],
    },
    "COLLECT": {
        "preconditions": ["Collectible resources in range"],
        "steps": [
            "Identify nearest valuable collectible",
            "Navigate to collectible",
            "Acquire and confirm collection",
        ],
        "success_criteria": ["Target resource collected"],
        "abort_criteria": ["Collection path too dangerous"],
        "failure_modes": ["Detour to collect cost more than the resource is worth"],
    },
    "BUILD": {
        "preconditions": ["Resources available for construction"],
        "steps": [
            "Select build target",
            "Place or construct elements",
            "Verify build is functional",
        ],
        "success_criteria": ["Construction completed"],
        "abort_criteria": ["Resources insufficient or location blocked"],
        "failure_modes": ["Build placed suboptimally, blocking future moves"],
    },
    "EXECUTE": {
        "preconditions": ["Action opportunity present"],
        "steps": [
            "Evaluate best available action",
            "Execute chosen action",
            "Observe result",
        ],
        "success_criteria": ["Action completed with positive effect"],
        "abort_criteria": ["No productive action available"],
        "failure_modes": ["No progress toward skill objective after several moves"],
    },
}


def _extract_tag_from_skill_id(skill_id: str) -> str:
    """Extract the subgoal tag from a compound skill ID like 'midgame:CLEAR'."""
    if ":" in skill_id:
        tag = skill_id.split(":", 1)[1].upper()
    else:
        tag = skill_id.upper()
    if tag in _SUBGOAL_TAG_SET:
        return tag
    return _TAG_ALIASES.get(tag, "EXECUTE")


def _extract_phase_from_skill_id(skill_id: str) -> str:
    """Extract the phase from a compound skill ID like 'midgame:CLEAR'."""
    if ":" in skill_id:
        return skill_id.split(":", 1)[0]
    return ""


def enrich_skill_protocols(
    agent: Any,
    segment_durations: Optional[Dict[str, List[int]]] = None,
) -> int:
    """Fill empty protocols on skills using tag-based templates.

    Mirrors ``populate_skill_protocols()`` in extract_skillbank_gpt54.py
    but uses deterministic templates instead of LLM calls for speed.

    Returns the number of skills updated.
    """
    from skill_agents_grpo.stage3_mvp.schemas import Protocol

    bank = agent.bank
    updated = 0

    for sid in list(bank.skill_ids):
        skill = bank.get_skill(sid)
        if skill is None or getattr(skill, "retired", False):
            continue
        if skill.protocol.steps and getattr(skill.protocol, "source", "template") == "llm":
            continue

        tag = _extract_tag_from_skill_id(sid)
        phase = _extract_phase_from_skill_id(sid)
        template = _TAG_PROTOCOL_TEMPLATES.get(tag, _TAG_PROTOCOL_TEMPLATES["EXECUTE"])

        preconditions = list(template["preconditions"])
        steps = list(template["steps"])
        success_criteria = list(template["success_criteria"])
        abort_criteria = list(template["abort_criteria"])

        contract = skill.contract
        if contract is not None:
            eff_add = getattr(contract, "eff_add", None) or set()
            eff_del = getattr(contract, "eff_del", None) or set()
            if eff_add:
                steps.append(f"Achieve: {', '.join(sorted(eff_add)[:3])}")
                success_criteria = [
                    f"{lit} achieved" for lit in sorted(eff_add)[:2]
                ] + success_criteria[:1]
            if eff_del:
                steps.append(f"Remove: {', '.join(sorted(eff_del)[:3])}")

        if phase:
            preconditions.insert(0, f"Game is in {phase} phase")

        durations = (segment_durations or {}).get(sid, [])
        if durations:
            avg_dur = max(1, sum(durations) // len(durations))
        else:
            avg_dur = 10

        protocol = Protocol(
            preconditions=preconditions,
            steps=steps[:7],
            success_criteria=success_criteria[:3],
            abort_criteria=abort_criteria[:3],
            expected_duration=avg_dur,
        )

        skill.protocol = protocol
        bank.add_or_update_skill(skill)
        updated += 1

    if updated:
        logger.info("Enriched %d skill(s) with tag-based protocols", updated)
    return updated


def enrich_execution_hints(agent: Any) -> int:
    """Generate ExecutionHint for skills that lack one.

    Mirrors ``_populate_execution_hints()`` in extract_skillbank_gpt54.py.
    """
    from skill_agents_grpo.stage3_mvp.schemas import ExecutionHint

    bank = agent.bank
    updated = 0

    for sid in list(bank.skill_ids):
        skill = bank.get_skill(sid)
        if skill is None or getattr(skill, "retired", False):
            continue
        if skill.execution_hint is not None:
            continue

        tag = _extract_tag_from_skill_id(sid)
        phase = _extract_phase_from_skill_id(sid)
        template = _TAG_PROTOCOL_TEMPLATES.get(tag, _TAG_PROTOCOL_TEMPLATES["EXECUTE"])

        name = skill.name or sid
        desc = skill.strategic_description or ""
        if not desc and skill.contract:
            eff_add = getattr(skill.contract, "eff_add", None) or set()
            eff_del = getattr(skill.contract, "eff_del", None) or set()
            parts = []
            if eff_add:
                parts.append("causes: " + ", ".join(sorted(eff_add)[:4]))
            if eff_del:
                parts.append("ends: " + ", ".join(sorted(eff_del)[:4]))
            desc = "; ".join(parts) if parts else name

        preconditions = skill.protocol.preconditions[:2] if skill.protocol.preconditions else []
        success_crit = skill.protocol.success_criteria[:2] if skill.protocol.success_criteria else []

        termination_cues = list(success_crit) if success_crit else []
        if not termination_cues and skill.contract:
            eff_add = getattr(skill.contract, "eff_add", None) or set()
            if eff_add:
                termination_cues = [f"{lit} achieved" for lit in sorted(eff_add)[:2]]
        if not termination_cues:
            termination_cues = [f"{name} objective met"]

        failure_modes = [template.get("failure_modes", ["No progress"])[0]]

        n_refs = len(skill.sub_episodes) if skill.sub_episodes else 0
        transition = f"[{tag}] {desc[:80]}" if tag else desc[:80]

        hint = ExecutionHint(
            common_preconditions=preconditions,
            common_target_objects=[],
            state_transition_pattern=transition,
            termination_cues=termination_cues,
            common_failure_modes=failure_modes,
            execution_description=desc[:150],
            n_source_segments=n_refs,
        )

        skill.execution_hint = hint
        bank.add_or_update_skill(skill)
        updated += 1

    if updated:
        logger.info("Generated %d execution hint(s)", updated)
    return updated


def compute_segment_durations(agent: Any) -> Dict[str, List[int]]:
    """Compute per-skill segment durations from accumulated segments."""
    durations: Dict[str, List[int]] = {}
    for seg in getattr(agent, "_all_segments", []):
        sid = getattr(seg, "skill_label", None)
        if not sid or sid in ("__NEW__", "NEW"):
            continue
        t_start = getattr(seg, "t_start", 0)
        t_end = getattr(seg, "t_end", 0)
        dur = max(1, t_end - t_start)
        durations.setdefault(sid, []).append(dur)
    return durations


def update_expected_durations(
    agent: Any,
    segment_durations: Dict[str, List[int]],
) -> int:
    """Update skill protocol expected_duration from actual segment data."""
    bank = agent.bank
    updated = 0
    for sid, durs in segment_durations.items():
        if not durs:
            continue
        skill = bank.get_skill(sid)
        if skill is None:
            continue
        avg = max(1, sum(durs) // len(durs))
        if skill.protocol.expected_duration != avg:
            skill.protocol.expected_duration = avg
            bank.add_or_update_skill(skill)
            updated += 1
    if updated:
        logger.info("Updated expected_duration for %d skill(s)", updated)
    return updated


def link_sub_episode_outcomes(
    agent: Any,
    episodes: list,
) -> int:
    """Create SubEpisodeRef entries and track success/failure outcomes.

    Mirrors ``_link_sub_episodes_to_skills()`` in extract_skillbank_gpt54.py.
    """
    from skill_agents_grpo.stage3_mvp.schemas import SubEpisodeRef

    bank = agent.bank
    linked = 0

    for seg in getattr(agent, "_all_segments", []):
        sid = getattr(seg, "skill_label", None)
        if not sid or sid in ("__NEW__", "NEW"):
            continue
        skill = bank.get_skill(sid)
        if skill is None:
            continue

        t_start = getattr(seg, "t_start", 0)
        t_end = getattr(seg, "t_end", 0)
        traj_id = getattr(seg, "traj_id", "")

        cum_reward = 0.0
        n_steps = max(1, t_end - t_start)
        obs_by_traj = getattr(agent, "_observations_by_traj", {})
        if traj_id in obs_by_traj:
            pass

        intent_tags = []
        for ep in episodes:
            exps = getattr(ep, "experiences", [])
            for t in range(t_start, min(t_end, len(exps))):
                exp = exps[t]
                r = getattr(exp, "reward", 0.0) or 0.0
                cum_reward += r
                intent = getattr(exp, "intentions", None) or ""
                m = _TAG_RE.match(str(intent).strip())
                if m:
                    intent_tags.append(m.group(1).upper())
            if cum_reward != 0.0:
                break

        outcome = "success" if cum_reward > 0 else "partial"

        ref = SubEpisodeRef(
            episode_id=getattr(seg, "episode_id", "") or traj_id,
            seg_start=t_start,
            seg_end=t_end,
            rollout_source=traj_id,
            summary=f"{sid}: {n_steps} steps, r={cum_reward:.1f}",
            intention_tags=intent_tags[:10],
            outcome=outcome,
            cumulative_reward=cum_reward,
        )

        if skill.sub_episodes is None:
            skill.sub_episodes = []
        skill.sub_episodes.append(ref)
        skill.n_instances = max(skill.n_instances, len(skill.sub_episodes))
        bank.add_or_update_skill(skill)
        linked += 1

    if linked:
        logger.info("Linked %d sub-episode ref(s) to skills", linked)
    return linked


def _enrich_role_side_stage_tags(
    agent: Any,
    episodes: Optional[list] = None,
) -> int:
    """Augment skill ``tags`` with role / side / stage from episode metadata.

    Scans sub-episode references and the episodes themselves to find
    role, side, and stage labels.  Adds them to ``skill.tags`` using
    canonical prefixes (``role:<name>``, ``side:<name>``, ``stage:<name>``)
    so the skill bank can segment and query skills along these dimensions.

    Only fires when episode metadata actually contains role info
    (i.e. ``unified_role_rollouts=True`` was used during rollout
    collection).  Otherwise this is a no-op.
    """
    bank = agent.bank
    updated = 0

    seg_to_meta: Dict[str, Dict[str, str]] = {}
    if episodes:
        for ep in episodes:
            ep_meta = getattr(ep, "metadata", {}) or {}
            ep_role = ep_meta.get("role", "")
            ep_side = ep_meta.get("side", "")
            if not ep_role:
                continue
            eid = getattr(ep, "episode_id", "")
            for exp in getattr(ep, "experiences", []):
                iface = getattr(exp, "interface", {}) or {}
                idx = getattr(exp, "idx", 0)
                key = f"{eid}:{idx}"
                seg_to_meta[key] = {
                    "role": iface.get("role", ep_role),
                    "side": iface.get("side", ep_side),
                    "stage": iface.get("stage", ""),
                }

    if not seg_to_meta:
        return 0

    for seg in getattr(agent, "_all_segments", []):
        sid = getattr(seg, "skill_label", None)
        if not sid or sid in ("__NEW__", "NEW"):
            continue
        skill = bank.get_skill(sid)
        if skill is None:
            continue

        traj_id = getattr(seg, "traj_id", "")
        t_start = getattr(seg, "t_start", 0)
        t_end = getattr(seg, "t_end", 0)

        roles_seen: Set[str] = set()
        sides_seen: Set[str] = set()
        stages_seen: Set[str] = set()
        for t in range(t_start, t_end):
            key = f"{traj_id}:{t}"
            meta = seg_to_meta.get(key, {})
            if meta.get("role"):
                roles_seen.add(meta["role"])
            if meta.get("side"):
                sides_seen.add(meta["side"])
            if meta.get("stage"):
                stages_seen.add(meta["stage"])

        new_tags: List[str] = []
        existing = set(skill.tags or [])
        for r in sorted(roles_seen):
            tag = f"role:{r}"
            if tag not in existing:
                new_tags.append(tag)
        for s in sorted(sides_seen):
            tag = f"side:{s}"
            if tag not in existing:
                new_tags.append(tag)
        for st in sorted(stages_seen):
            tag = f"stage:{st}"
            if tag not in existing:
                new_tags.append(tag)

        if new_tags:
            skill.tags = list(existing | set(new_tags))
            bank.add_or_update_skill(skill)
            updated += 1

    if updated:
        logger.info(
            "Enriched %d skill(s) with role/side/stage tags", updated,
        )
    return updated


def enrich_bank_after_update(
    agent: Any,
    episodes: Optional[list] = None,
) -> Dict[str, int]:
    """Run all enrichment steps after a bank update.

    Call this after Stage 3+4 in the co-evolution pipeline.
    Returns a dict of counts for each enrichment type.
    """
    results: Dict[str, int] = {}

    durations = compute_segment_durations(agent)
    results["protocols"] = enrich_skill_protocols(agent, segment_durations=durations)
    results["execution_hints"] = enrich_execution_hints(agent)
    results["durations_updated"] = update_expected_durations(agent, durations)

    if episodes:
        results["sub_episode_refs"] = link_sub_episode_outcomes(agent, episodes)
        results["role_side_stage_tags"] = _enrich_role_side_stage_tags(
            agent, episodes,
        )

    total = sum(results.values())
    if total:
        logger.info(
            "Skill enrichment: %d protocol(s), %d hint(s), %d duration(s), "
            "%d ref(s), %d role/side/stage tag(s)",
            results.get("protocols", 0),
            results.get("execution_hints", 0),
            results.get("durations_updated", 0),
            results.get("sub_episode_refs", 0),
            results.get("role_side_stage_tags", 0),
        )
    return results
