"""
Tool-call reward for agentic RL training.

Computes a reward signal for the decision agent's tool calls (query_skill,
query_memory, call_skill) so that RL can learn when and how to use tools.
Used alongside env reward and shaping in decision_agents.reward_func.

Usage::

    from skill_agents.tool_call_reward import compute_tool_call_reward, ToolCallRewardConfig

    reward = compute_tool_call_reward(
        tool_name="query_skill",
        tool_args={"key": "navigate to pot and place onion"},
        context_observation="chef near pot, holding onion",
        outcome_observation="onion in pot, soup cooking",
        skill_bank=bank,
        retrieved_skill_id="nav_to_pot",
        config=ToolCallRewardConfig(),
    )
    # reward["r_total"] for RL loss / value target
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# Optional: use query engine for relevance scoring
try:
    from skill_agents.query import SkillQueryEngine
except ImportError:
    SkillQueryEngine = None

try:
    from skill_agents.skill_bank.bank import SkillBankMVP
except ImportError:
    SkillBankMVP = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRewardConfig:
    """Weights and scales for tool-call reward (agentic RL)."""

    # Relevance: how well the tool call matched the context / retrieval quality.
    w_relevance: float = 1.0
    # Utility: how much the outcome satisfied the tool's goal (e.g. skill effects).
    w_utility: float = 1.0
    # Scale retrieval score [0,1] to reward (e.g. 0.5 so max relevance reward = 0.5).
    relevance_scale: float = 0.5
    # Per-predicate satisfaction bonus in outcome (utility).
    utility_per_predicate: float = 0.1
    # Bonus when all skill effects are satisfied in outcome.
    utility_full_completion: float = 0.3
    # When we have no engine/bank, default reward for query_skill (e.g. small positive to avoid killing exploration).
    default_query_reward: float = 0.0
    # query_memory: if caller does not pass retrieval_quality, use this.
    default_memory_reward: float = 0.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRewardResult:
    """Reward components for one tool call (for RL training)."""

    r_relevance: float = 0.0
    r_utility: float = 0.0
    r_total: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r_relevance": self.r_relevance,
            "r_utility": self.r_utility,
            "r_total": self.r_total,
            **self.details,
        }


def _tokenize(text: str) -> Set[str]:
    return {w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(w) >= 2}


def _predicates_satisfied_in_text(predicates: Set[str], text: str) -> Set[str]:
    """Return the subset of predicates whose tokens appear in text."""
    if not text or not predicates:
        return set()
    text_lower = text.lower()
    satisfied: Set[str] = set()
    for pred in predicates:
        tokens = _tokenize(pred)
        if tokens and all(t in text_lower for t in tokens):
            satisfied.add(pred)
    return satisfied


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_tool_call_reward(
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
    context_observation: Optional[str] = None,
    outcome_observation: Optional[str] = None,
    skill_bank: Any = None,
    query_engine: Any = None,
    retrieved_skill_id: Optional[str] = None,
    retrieved_result: Optional[Dict[str, Any]] = None,
    skill_contract: Any = None,
    retrieval_quality: Optional[float] = None,
    config: Optional[ToolCallRewardConfig] = None,
) -> ToolCallRewardResult:
    """
    Compute reward for a single tool call, for agentic RL training.

    Args:
        tool_name: One of "query_skill", "query_memory", "take_action" / "call_skill".
        tool_args: Args passed to the tool (e.g. {"key": "..."} for query_skill).
        context_observation: State/observation before the tool call.
        outcome_observation: State/observation after (or outcome summary).
        skill_bank: SkillBankMVP (or agent with .bank) for relevance/utility.
        query_engine: Optional SkillQueryEngine; if None and skill_bank is SkillBankMVP, one is built.
        retrieved_skill_id: Skill ID that was retrieved (for query_skill) or called (for call_skill).
        retrieved_result: Full result from query_skill (e.g. {"skill_id", "score", "micro_plan"}).
        skill_contract: SkillEffectsContract (or dict with eff_add) for utility from effects.
        retrieval_quality: Optional [0,1] quality for query_memory when not computed here.
        config: Weights and scales; default ToolCallRewardConfig().

    Returns:
        ToolCallRewardResult with r_relevance, r_utility, r_total and optional details.
    """
    cfg = config or ToolCallRewardConfig()
    tool_args = tool_args or {}
    details: Dict[str, Any] = {}

    # Resolve bank and engine
    bank = skill_bank
    if bank is not None and getattr(bank, "bank", None) is not None:
        bank = getattr(bank, "bank", bank)
    engine = query_engine
    if engine is None and bank is not None and SkillQueryEngine is not None:
        try:
            engine = SkillQueryEngine(bank)
        except Exception:
            engine = None

    # ----- query_skill -----
    if tool_name == "query_skill":
        key = tool_args.get("key", "")
        skill_id = retrieved_skill_id or (retrieved_result or {}).get("skill_id")
        score = (retrieved_result or {}).get("score")

        # Relevance: retrieval score from engine, or from retrieved_result
        if score is not None and isinstance(score, (int, float)):
            r_relevance = float(score) * cfg.relevance_scale
        elif engine is not None and key:
            try:
                results = engine.query(key, top_k=1)
                if results:
                    r_relevance = float(results[0].get("score", 0.0)) * cfg.relevance_scale
                    if skill_id is None:
                        skill_id = results[0].get("skill_id")
                else:
                    r_relevance = cfg.default_query_reward
            except Exception:
                r_relevance = cfg.default_query_reward
        else:
            r_relevance = cfg.default_query_reward

        # Utility: does outcome satisfy the retrieved skill's effects?
        contract = skill_contract
        if contract is None and skill_id and bank is not None:
            try:
                contract = bank.get_contract(skill_id)
            except Exception:
                contract = None
        eff_add: Set[str] = set()
        if contract is not None:
            eff_add = getattr(contract, "eff_add", set()) or set()
        if outcome_observation and eff_add:
            satisfied = _predicates_satisfied_in_text(eff_add, outcome_observation)
            n_sat = len(satisfied)
            r_utility = n_sat * cfg.utility_per_predicate
            if satisfied == eff_add:
                r_utility += cfg.utility_full_completion
            details["satisfied_predicates"] = list(satisfied)
            details["total_predicates"] = len(eff_add)
        else:
            r_utility = 0.0

        r_total = cfg.w_relevance * r_relevance + cfg.w_utility * r_utility
        return ToolCallRewardResult(
            r_relevance=r_relevance,
            r_utility=r_utility,
            r_total=r_total,
            details={**details, "tool": "query_skill", "skill_id": skill_id},
        )

    # ----- query_memory -----
    if tool_name == "query_memory":
        if retrieval_quality is not None:
            r_relevance = float(retrieval_quality) * cfg.relevance_scale
        else:
            r_relevance = cfg.default_memory_reward
        r_utility = 0.0
        r_total = cfg.w_relevance * r_relevance
        return ToolCallRewardResult(
            r_relevance=r_relevance,
            r_utility=r_utility,
            r_total=r_total,
            details={"tool": "query_memory"},
        )

    # ----- call_skill / take_action (when following a skill) -----
    if tool_name in ("call_skill", "take_action") and (retrieved_skill_id or skill_contract):
        contract = skill_contract
        if contract is None and retrieved_skill_id and bank is not None:
            try:
                contract = bank.get_contract(retrieved_skill_id)
            except Exception:
                contract = None
        eff_add = set()
        if contract is not None:
            eff_add = getattr(contract, "eff_add", set()) or set()
        if outcome_observation and eff_add:
            satisfied = _predicates_satisfied_in_text(eff_add, outcome_observation)
            n_sat = len(satisfied)
            r_utility = n_sat * cfg.utility_per_predicate
            if satisfied == eff_add:
                r_utility += cfg.utility_full_completion
            details["satisfied_predicates"] = list(satisfied)
        else:
            r_utility = 0.0
        r_relevance = 0.0
        r_total = cfg.w_utility * r_utility
        return ToolCallRewardResult(
            r_relevance=r_relevance,
            r_utility=r_utility,
            r_total=r_total,
            details={**details, "tool": tool_name, "skill_id": retrieved_skill_id},
        )

    # ----- primitive take_action (no skill) / unknown -----
    return ToolCallRewardResult(
        r_relevance=0.0,
        r_utility=0.0,
        r_total=0.0,
        details={"tool": tool_name},
    )


def compute_episode_tool_call_returns(
    tool_call_trajectory: List[Dict[str, Any]],
    skill_bank: Any = None,
    query_engine: Any = None,
    config: Optional[ToolCallRewardConfig] = None,
    gamma: float = 0.99,
) -> List[float]:
    """
    Compute per-step tool-call rewards and optionally discount to returns for RL.

    tool_call_trajectory: list of dicts, each with at least:
      - tool_name, tool_args
      - optional: context_observation, outcome_observation, retrieved_skill_id, retrieved_result, skill_contract

    Returns list of r_total per step (same length as trajectory).
    """
    cfg = config or ToolCallRewardConfig()
    rewards: List[float] = []
    for step in tool_call_trajectory:
        rr = compute_tool_call_reward(
            tool_name=step.get("tool_name", "take_action"),
            tool_args=step.get("tool_args"),
            context_observation=step.get("context_observation"),
            outcome_observation=step.get("outcome_observation"),
            skill_bank=skill_bank,
            query_engine=query_engine,
            retrieved_skill_id=step.get("retrieved_skill_id"),
            retrieved_result=step.get("retrieved_result"),
            skill_contract=step.get("skill_contract"),
            retrieval_quality=step.get("retrieval_quality"),
            config=cfg,
        )
        rewards.append(rr.r_total)
    return rewards
