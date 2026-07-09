"""
Reward shaping for the Decision Agent, compatible with VERL.

Two interfaces:
  1. TrainRewardShaper — original per-step reward computation (used inside
     the environment workers during rollout collection).
  2. GameAIRewardManager — VERL reward manager that converts episode-level
     scalar rewards into per-token reward tensors for advantage computation.

VERL integration:
  The environment workers (GameAIVecEnv) already accumulate shaped rewards
  (r_env + r_follow + r_cost) into the per-step rewards returned by step().
  VERL's TrajectoryCollector sums these into episode_rewards.
  GameAIRewardManager then converts that scalar into a token-level tensor
  by placing it on the last response token, as required by VERL's
  advantage estimators (GRPO, GiGPO, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardBreakdown:
    """Extended reward breakdown for training logs."""
    r_env: float = 0.0
    r_follow: float = 0.0
    r_cost: float = 0.0
    r_tool: float = 0.0
    r_total: float = 0.0
    action_type: str = "primitive"
    active_skill_id: Optional[str] = None
    newly_satisfied: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r_env": self.r_env, "r_follow": self.r_follow,
            "r_cost": self.r_cost, "r_tool": self.r_tool,
            "r_total": self.r_total,
            "action_type": self.action_type,
            "active_skill_id": self.active_skill_id,
            "newly_satisfied": self.newly_satisfied,
        }


class TrainRewardShaper:
    """Per-step reward shaper (used inside environment workers).

    Wraps decision_agents.reward_func.RewardComputer with bank-state-aware
    shaping and tool-call reward integration from skill_agents.
    """

    def __init__(self, config=None, skill_bank=None, tool_call_reward_weight: float = 0.1):
        try:
            from decision_agents.reward_func import RewardComputer, RewardConfig
            self.config = config or RewardConfig()
            self.computer = RewardComputer(self.config)
        except ImportError:
            self.config = config
            self.computer = None
        self.skill_bank = skill_bank
        self.tool_call_reward_weight = tool_call_reward_weight

        self._tool_reward_fn = None
        self._tool_reward_cfg = None
        try:
            from skill_agents.tool_call_reward import (
                compute_tool_call_reward as _fn,
                ToolCallRewardConfig,
            )
            self._tool_reward_fn = _fn
            self._tool_reward_cfg = ToolCallRewardConfig()
        except ImportError:
            pass

    def reset(self) -> None:
        if self.computer is not None:
            self.computer.reset()

    def compute_reward(
        self, r_env: float, action_type: str, observation: str,
        active_skill_id: Optional[str] = None, skill_contract=None,
        bank_state=None,
        query_key: Optional[str] = None,
        outcome_observation: Optional[str] = None,
        retrieval_score: Optional[float] = None,
    ) -> RewardBreakdown:
        """Compute reward with full breakdown including tool-call reward."""
        if self.computer is None:
            r_tool = self._compute_tool_reward(
                action_type, query_key, active_skill_id,
                observation, outcome_observation, bank_state,
                retrieval_score,
            )
            r_total = r_env + self.tool_call_reward_weight * r_tool
            return RewardBreakdown(
                r_env=r_env, r_tool=r_tool, r_total=r_total,
                action_type=action_type,
            )

        bank = bank_state or self.skill_bank
        contract = skill_contract
        if contract is None and active_skill_id and bank:
            try:
                contract = bank.get_contract(active_skill_id)
            except Exception:
                contract = None

        prev_satisfied = len(self.computer.satisfied_predicates)
        rr = self.computer.compute_reward(
            r_env=r_env, action_type=action_type, observation=observation,
            active_skill_id=active_skill_id, skill_contract=contract,
        )
        new_satisfied = len(self.computer.satisfied_predicates) - prev_satisfied

        r_tool = self._compute_tool_reward(
            action_type, query_key, active_skill_id,
            observation, outcome_observation, bank,
            retrieval_score,
        )

        r_total = rr.r_total + self.tool_call_reward_weight * r_tool

        return RewardBreakdown(
            r_env=rr.r_env, r_follow=rr.r_follow, r_cost=rr.r_cost,
            r_tool=r_tool, r_total=r_total, action_type=action_type,
            active_skill_id=active_skill_id,
            newly_satisfied=max(new_satisfied, 0),
        )

    def _compute_tool_reward(
        self,
        action_type: str,
        query_key: Optional[str],
        active_skill_id: Optional[str],
        context_obs: Optional[str],
        outcome_obs: Optional[str],
        bank: Any,
        retrieval_score: Optional[float],
    ) -> float:
        """Compute tool-call reward using skill_agents.tool_call_reward."""
        if self._tool_reward_fn is None:
            return 0.0
        if action_type not in ("QUERY_SKILL", "QUERY_MEM", "CALL_SKILL"):
            return 0.0

        tool_name_map = {
            "QUERY_SKILL": "query_skill",
            "QUERY_MEM": "query_memory",
            "CALL_SKILL": "call_skill",
        }
        tool_name = tool_name_map.get(action_type, "take_action")

        try:
            result = self._tool_reward_fn(
                tool_name=tool_name,
                tool_args={"key": query_key or ""},
                context_observation=context_obs,
                outcome_observation=outcome_obs,
                skill_bank=bank,
                retrieved_skill_id=active_skill_id,
                retrieval_quality=retrieval_score,
                config=self._tool_reward_cfg,
            )
            return result.r_total
        except Exception:
            return 0.0

    def update_bank(self, new_bank) -> None:
        self.skill_bank = new_bank

    @property
    def cumulative(self):
        return self.computer.cumulative if self.computer else None

    @property
    def history(self) -> list:
        return self.computer.history if self.computer else []


# ---------------------------------------------------------------------------
# VERL-compatible reward manager
# ---------------------------------------------------------------------------
try:
    import torch
    from verl import DataProto
    _HAS_VERL = True
except ImportError:
    _HAS_VERL = False


if _HAS_VERL:
    class GameAIRewardManager:
        """VERL reward manager for Game-AI.

        Converts episode-level scalar rewards (already shaped by the env
        workers) into per-token reward tensors as required by VERL's
        advantage computation pipeline.
        """

        def __init__(
            self, tokenizer, num_examine: int = 0,
            normalize_by_length: bool = False,
        ):
            self.tokenizer = tokenizer
            self.num_examine = num_examine
            self.normalize_by_length = normalize_by_length

        def __call__(self, data: DataProto, return_dict: bool = False):
            if "rm_scores" in data.batch.keys():
                if return_dict:
                    return {"reward_tensor": data.batch["rm_scores"]}
                return data.batch["rm_scores"]

            reward_tensor = torch.zeros_like(
                data.batch["responses"], dtype=torch.float32
            )
            already_printed = {}

            for i in range(len(data)):
                data_item = data[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

                episode_rewards = data_item.non_tensor_batch["episode_rewards"]
                episode_lengths = data_item.non_tensor_batch["episode_lengths"]

                if self.normalize_by_length and episode_lengths > 0:
                    score = episode_rewards / episode_lengths
                else:
                    score = episode_rewards

                reward_tensor[i, valid_response_length - 1] = torch.tensor(
                    score, dtype=torch.float32, device=prompt_ids.device
                )

                data_source = data_item.non_tensor_batch.get("data_source", "gameai")
                if data_source not in already_printed:
                    already_printed[data_source] = 0
                if (already_printed[data_source] < self.num_examine
                        and np.random.random() < 0.1):
                    already_printed[data_source] += 1
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    p_str = self.tokenizer.decode(
                        prompt_ids[-valid_prompt_length:], skip_special_tokens=False
                    )
                    r_ids = data_item.batch["responses"][:valid_response_length]
                    r_str = self.tokenizer.decode(r_ids, skip_special_tokens=False)
                    print(f"[{data_source}][prompt] {p_str[:200]}")
                    print(f"[{data_source}][response] {r_str[:200]}")
                    print(f"[{data_source}][score] {score}")

            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            return reward_tensor
