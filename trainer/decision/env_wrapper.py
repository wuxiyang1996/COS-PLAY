"""
VERL-compatible environment wrapper for the Game-AI decision agent.

Implements verl-agent's EnvironmentManagerBase interface so that the
decision agent (with retrieval-as-action) can be trained via VERL's
RayPPOTrainer and multi-turn rollout loop.

VERL contract:
  - reset(kwargs) -> (observations_dict, infos_list)
  - step(text_actions) -> (observations_dict, rewards, dones, infos_list)
  - observations_dict = {"text": List[str], "image": ..., "anchor": ...}
  - infos[i] must contain "won" (bool) and "is_action_valid" (np.ndarray)

Also retains the original single-env EnvWrapper API for non-VERL use
(evaluation, debugging).
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------
RETRIEVAL_ACTIONS = {"QUERY_MEM", "QUERY_SKILL", "CALL_SKILL"}
_ACTION_RE = re.compile(
    r"(QUERY_MEM|QUERY_SKILL|CALL_SKILL)\s*\(\s*(.+?)\s*\)", re.IGNORECASE
)
_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_action(action: str) -> Dict[str, Any]:
    """Parse an action string into type + arguments."""
    action_stripped = (action or "").strip()
    match = _ACTION_RE.match(action_stripped)
    if match:
        atype = match.group(1).upper()
        raw_args = match.group(2).strip()
        if atype == "CALL_SKILL":
            parts = [p.strip().strip("'\"") for p in raw_args.split(",", 1)]
            return {"type": atype, "skill_id": parts[0],
                    "params": parts[1] if len(parts) > 1 else ""}
        return {"type": atype, "key": raw_args.strip("'\"") }

    action_upper = action_stripped.upper()
    for prefix in RETRIEVAL_ACTIONS:
        if action_upper.startswith(prefix):
            remainder = action_stripped[len(prefix):].strip("() '\"")
            if prefix == "CALL_SKILL":
                parts = [p.strip().strip("'\"") for p in remainder.split(",", 1)]
                return {"type": prefix, "skill_id": parts[0],
                        "params": parts[1] if len(parts) > 1 else ""}
            return {"type": prefix, "key": remainder}

    return {"type": "primitive", "action": action_stripped}


def gameai_projection(text_actions: List[str]) -> Tuple[List[str], List[int]]:
    """VERL projection function: extract action from LLM output.

    Expects <think>...</think><action>...</action> format.
    Retrieval actions are passed through to the env as-is.
    """
    parsed_actions = []
    valids = []
    for raw in text_actions:
        raw = raw.strip()
        match = _ACTION_TAG_RE.search(raw)
        if match:
            action_str = match.group(1).strip()
            if action_str:
                parsed_actions.append(action_str)
                valids.append(1)
            else:
                parsed_actions.append(raw[-50:] if len(raw) > 50 else raw)
                valids.append(0)
        else:
            clean = _THINK_TAG_RE.sub("", raw).strip()
            parsed_actions.append(clean[:200] if clean else "noop")
            valids.append(0)
    return parsed_actions, valids


# ---------------------------------------------------------------------------
# Per-env state
# ---------------------------------------------------------------------------
@dataclass
class ToolCallEvent:
    """Record of a single tool call made during a rollout step."""
    t: int = 0
    action_type: str = ""
    action_raw: str = ""
    key: Optional[str] = None
    skill_id: Optional[str] = None
    retrieved_ids: List[str] = field(default_factory=list)
    retrieval_score: float = 0.0


@dataclass
class WrapperState:
    """Internal state maintained per environment instance."""
    active_skill_id: Optional[str] = None
    active_skill_contract: Any = None
    prev_skill_id: Optional[str] = None
    retrieved_cards: List[Dict[str, Any]] = field(default_factory=list)
    memory_cards: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    episode_id: str = ""
    context_buffer: str = ""
    tool_call_trace: List[ToolCallEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vectorized env wrapping N game instances + skill bank + memory
# ---------------------------------------------------------------------------
class GameAIVecEnv:
    """Vectorized game-AI env with retrieval-as-action support.

    Each sub-environment wraps a game env and intercepts retrieval
    actions (QUERY_MEM, QUERY_SKILL, CALL_SKILL) before they reach
    the underlying game.  Consumed by GameAIEnvironmentManager.
    """

    def __init__(
        self,
        env_factory,
        env_num: int,
        group_n: int = 1,
        base_seed: int = 0,
        skill_bank=None,
        memory=None,
    ):
        self.num_envs = env_num * group_n
        self.group_n = group_n
        self.skill_bank = skill_bank
        self.memory = memory
        self._query_engine = None

        self.envs = []
        self.states: List[WrapperState] = []
        for i in range(self.num_envs):
            seed = base_seed + (i // group_n)
            env = env_factory(seed=seed)
            self.envs.append(env)
            self.states.append(WrapperState())

    def _get_query_engine(self):
        if self._query_engine is None and self.skill_bank is not None:
            try:
                from skill_agents.query import SkillQueryEngine
                self._query_engine = SkillQueryEngine(self.skill_bank)
            except ImportError:
                pass
        return self._query_engine

    def reset(self):
        text_obs, infos = [], []
        for i, env in enumerate(self.envs):
            self.states[i] = WrapperState()
            obs, info = env.reset()
            obs_text = str(obs) if not isinstance(obs, str) else obs
            self.states[i].context_buffer = obs_text
            if not isinstance(info, dict):
                info = {}
            info["won"] = False
            info["action_type"] = "reset"
            info["active_skill_id"] = None
            text_obs.append(obs_text)
            infos.append(info)
        return text_obs, infos

    def step(self, actions: List[str]):
        assert len(actions) == self.num_envs
        text_obs, rewards, dones, infos = [], [], [], []
        for i, action in enumerate(actions):
            st = self.states[i]
            st.step_count += 1
            st.prev_skill_id = st.active_skill_id

            parsed = parse_action(action)
            atype = parsed["type"]

            if atype in RETRIEVAL_ACTIONS:
                o, r, d, info = self._handle_retrieval(i, parsed)
            else:
                o, r, d, info = self._handle_primitive(i, action)

            text_obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return text_obs, rewards, dones, infos

    def _handle_retrieval(self, idx, parsed):
        st = self.states[idx]
        atype = parsed["type"]
        retrieval_score = 0.0
        retrieved_ids: List[str] = []

        if atype == "QUERY_SKILL" and self.skill_bank is not None:
            key = parsed.get("key", "")
            engine = self._get_query_engine()
            retrieved = []
            if engine is not None:
                retrieved = engine.query(key, top_k=3)
            elif hasattr(self.skill_bank, "skill_ids"):
                retrieved = [{"skill_id": sid} for sid in self.skill_bank.skill_ids[:3]]
            st.retrieved_cards = retrieved
            retrieved_ids = [c.get("skill_id", "") for c in retrieved]
            if retrieved:
                retrieval_score = float(retrieved[0].get("score", 0.0))
            card_text = "\n".join(
                f"  [{c.get('skill_id', '?')}] score={c.get('score', 0):.3f}"
                for c in retrieved
            )
            st.context_buffer += f"\n[QUERY_SKILL result]\n{card_text}"

        elif atype == "QUERY_MEM" and self.memory is not None:
            key = parsed.get("key", "")
            results = self.memory.query(key, k=3)
            st.memory_cards = results if isinstance(results, list) else []
            st.context_buffer += f"\n[QUERY_MEM result]\n{str(results)[:500]}"

        elif atype == "CALL_SKILL":
            skill_id = parsed.get("skill_id", "")
            st.active_skill_id = skill_id
            st.active_skill_contract = None
            if self.skill_bank and hasattr(self.skill_bank, "get_contract"):
                st.active_skill_contract = self.skill_bank.get_contract(skill_id)
            retrieved_ids = [skill_id]
            st.context_buffer += f"\n[CALL_SKILL activated: {skill_id}]"

        event = ToolCallEvent(
            t=st.step_count,
            action_type=atype,
            action_raw=parsed.get("key", parsed.get("skill_id", "")),
            key=parsed.get("key"),
            skill_id=parsed.get("skill_id") or (retrieved_ids[0] if retrieved_ids else None),
            retrieved_ids=retrieved_ids,
            retrieval_score=retrieval_score,
        )
        st.tool_call_trace.append(event)

        info = {
            "won": False, "action_type": atype,
            "active_skill_id": st.active_skill_id,
            "prev_skill_id": st.prev_skill_id,
            "query_key": parsed.get("key"),
            "retrieval_score": retrieval_score,
            "retrieved_ids": retrieved_ids,
        }
        return st.context_buffer, 0.0, False, info

    def _handle_primitive(self, idx, action):
        env = self.envs[idx]
        st = self.states[idx]
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        r_env = float(reward) if not isinstance(reward, dict) else sum(reward.values())
        obs_text = str(obs) if not isinstance(obs, str) else obs
        st.context_buffer = obs_text
        if not isinstance(info, dict):
            info = {}
        info["won"] = info.get("won", done and r_env > 0)
        info["action_type"] = "primitive"
        info["active_skill_id"] = st.active_skill_id
        info["prev_skill_id"] = st.prev_skill_id
        info["query_key"] = None
        return obs_text, r_env, done, info

    def get_tool_call_traces(self) -> List[List[ToolCallEvent]]:
        """Return per-env tool call traces for the current episodes."""
        return [list(st.tool_call_trace) for st in self.states]

    def get_episode_tool_calls(self, env_idx: int) -> List[Dict[str, Any]]:
        """Return tool call dicts for a single environment's current episode."""
        if env_idx >= len(self.states):
            return []
        return [
            {
                "t": e.t, "action_type": e.action_type,
                "key": e.key, "skill_id": e.skill_id,
                "retrieved_ids": e.retrieved_ids,
                "retrieval_score": e.retrieval_score,
            }
            for e in self.states[env_idx].tool_call_trace
        ]

    def update_skill_bank(self, new_bank):
        """Hot-swap the skill bank (called after co-evolution bank update)."""
        self.skill_bank = new_bank
        self._query_engine = None
        for st in self.states:
            st.retrieved_cards = []

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()


# ---------------------------------------------------------------------------
# VERL EnvironmentManagerBase implementation
# ---------------------------------------------------------------------------
try:
    from agent_system.environments.base import EnvironmentManagerBase, to_numpy
    from agent_system.memory import SimpleMemory
    _HAS_VERL = True
except ImportError:
    _HAS_VERL = False
    to_numpy = lambda x: np.array(x)

# Prompt templates
SYSTEM_PROMPT = """\
You are a game-playing AI agent. At each step you observe the game state and \
choose ONE action. You can also use retrieval tools:
  QUERY_SKILL(key) — search the skill bank for relevant skills.
  QUERY_MEM(key) — search episodic memory for similar past situations.
  CALL_SKILL(skill_id) — activate a skill from the bank.

Output format:
<think>Your reasoning</think>
<action>YOUR_ACTION</action>"""

TEMPLATE_NO_HIS = "{system_prompt}\n\n## Current Observation\n{current_observation}"
TEMPLATE_WITH_HIS = (
    "{system_prompt}\n\n## Step {step_count} (last {history_length} shown)\n\n"
    "## Recent History\n{action_history}\n\n"
    "## Current Observation (Step {current_step})\n{current_observation}"
)


if _HAS_VERL:
    class GameAIEnvironmentManager(EnvironmentManagerBase):
        """VERL environment manager for the Game-AI decision agent.

        Plugs into verl-agent's make_envs() and TrajectoryCollector to
        provide text-only observations with retrieval action support.
        """

        def __init__(self, envs: GameAIVecEnv, projection_f, config):
            self.memory = SimpleMemory()
            super().__init__(envs, projection_f, config)

        def reset(self, kwargs=None):
            text_obs, infos = self.envs.reset()
            self.memory.reset(batch_size=len(text_obs))
            self.pre_text_obs = text_obs
            full_text_obs = self.build_text_obs(text_obs, init=True)
            return {"text": full_text_obs, "image": None, "anchor": text_obs}, infos

        def step(self, text_actions: List[str]):
            actions, valids = self.projection_f(text_actions)
            text_obs, rewards, dones, infos = self.envs.step(actions)
            self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
            self.pre_text_obs = text_obs
            full_text_obs = self.build_text_obs(text_obs)
            for i, info in enumerate(infos):
                info["is_action_valid"] = to_numpy(valids[i])
            return (
                {"text": full_text_obs, "image": None, "anchor": text_obs},
                to_numpy(rewards),
                to_numpy(dones),
                infos,
            )

        def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
            history_length = getattr(self.config.env, "history_length", 5)
            out = []
            if not init and history_length > 0:
                memory_contexts, valid_lens = self.memory.fetch(
                    history_length, obs_key="text_obs", action_key="action",
                )
            for i in range(len(text_obs)):
                if init or history_length <= 0:
                    obs = TEMPLATE_NO_HIS.format(
                        system_prompt=SYSTEM_PROMPT,
                        current_observation=text_obs[i],
                    )
                else:
                    obs = TEMPLATE_WITH_HIS.format(
                        system_prompt=SYSTEM_PROMPT,
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                    max_len = getattr(self.config.env, "max_obs_length", 8000)
                    if len(obs) > max_len:
                        obs = TEMPLATE_NO_HIS.format(
                            system_prompt=SYSTEM_PROMPT,
                            current_observation=text_obs[i],
                        )
                out.append(obs)
            return out

        def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
            for i in reversed(range(len(total_batch_list[batch_idx]))):
                batch_item = total_batch_list[batch_idx][i]
                if batch_item["active_masks"]:
                    info = total_infos[batch_idx][i]
                    success["success_rate"].append(float(info.get("won", 0)))
                    return

        def update_skill_bank(self, new_bank):
            if hasattr(self.envs, "update_skill_bank"):
                self.envs.update_skill_bank(new_bank)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_gameai_envs(
    env_factory, seed: int, env_num: int, group_n: int = 1,
    skill_bank=None, memory=None,
) -> GameAIVecEnv:
    """Build vectorized game-AI environments."""
    return GameAIVecEnv(
        env_factory=env_factory, env_num=env_num,
        group_n=group_n, base_seed=seed,
        skill_bank=skill_bank, memory=memory,
    )
