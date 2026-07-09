# VLM decision-making agent: Summary → Select Skill → Act → Update Intention → Reward → Repeat.
# Tools: take_action, get_state_summary, get_intention, select_skill, reward.
# Required each step: get_state_summary (before action), take_action, get_intention (after action), reward.
# select_skill is the ONLY protocol to query prior rollouts, experience, and skill plans.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

from data_structure.experience import Experience, Episode

from .agent_helper import (
    get_state_summary as helper_get_state_summary,
    infer_intention as helper_infer_intention,
    skill_bank_to_text,
    select_skill_from_bank,
    EpisodicMemoryStore,
    HARD_SUMMARY_CHAR_LIMIT,
)
from .reward_func import RewardComputer, RewardConfig, RewardResult

from .dummy_agent import (
    detect_game,
    extract_action,
    _default_action,
    GAME_OVERCOOKED,
    GAME_GAMINGAGENT,
    GAME_VIDEOGAMEBENCH,
    GAME_VIDEOGAMEBENCH_DOS,
    OVERCOOKED_VALID_ACTIONS,
    VIDEOGAMEBENCH_VALID_ACTIONS,
    VIDEOGAMEBENCH_DOS_VALID_KEYS,
)


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Internal state maintained by the VLM decision agent."""
    current_intention: str = ""
    progress_notes: List[str] = field(default_factory=list)
    last_actions: List[Any] = field(default_factory=list)
    stuck_counter: int = 0
    steps_since_retrieval: int = 0
    active_skill_plan: Optional[List[Dict[str, Any]]] = None
    active_skill_id: Optional[str] = None
    active_skill_guidance: Optional[Dict[str, Any]] = None
    skill_step_index: int = 0
    steps_without_progress: int = 0
    last_state_summary: str = ""
    last_reward: Optional[RewardResult] = None


# ---------------------------------------------------------------------------
# Tool names and defaults
# ---------------------------------------------------------------------------

TOOL_TAKE_ACTION = "take_action"
TOOL_GET_STATE_SUMMARY = "get_state_summary"
TOOL_GET_INTENTION = "get_intention"
TOOL_SELECT_SKILL = "select_skill"
TOOL_REWARD = "reward"

TOOLS = [TOOL_TAKE_ACTION, TOOL_GET_STATE_SUMMARY, TOOL_GET_INTENTION, TOOL_SELECT_SKILL, TOOL_REWARD]

DEFAULT_RETRIEVAL_BUDGET_N = 10
DEFAULT_SKILL_ABORT_K = 5
MAX_LAST_ACTIONS = 5
MAX_PROGRESS_NOTES = 3


# ---------------------------------------------------------------------------
# VLM Decision Agent
# ---------------------------------------------------------------------------

class VLMDecisionAgent:
    """
    VLM decision-making agent that uses skills as the primary decision unit.

    select_skill is the single protocol to query prior rollouts, experience,
    and skill plans.  The agent gets the full structured guidance package
    (protocol steps, preconditions, termination hints, failure modes) and
    follows the protocol to execute.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: Optional[str] = None,
        skill_bank: Any = None,
        memory: Optional[EpisodicMemoryStore] = None,
        retrieval_budget_n: int = DEFAULT_RETRIEVAL_BUDGET_N,
        skill_abort_k: int = DEFAULT_SKILL_ABORT_K,
        game: Optional[str] = None,
        reward_config: Optional[RewardConfig] = None,
        embedder: Any = None,
    ) -> None:
        self.model = model or self.DEFAULT_MODEL
        self.skill_bank = skill_bank
        if memory is not None:
            self.memory = memory
        else:
            emb = embedder
            if emb is None:
                try:
                    from rag import get_text_embedder
                    emb = get_text_embedder()
                except Exception:
                    emb = None
            self.memory = EpisodicMemoryStore(embedder=emb)
        self.retrieval_budget_n = retrieval_budget_n
        self.skill_abort_k = skill_abort_k
        self.game_hint = game
        self.reward_config = reward_config or RewardConfig()
        self.reward_computer = RewardComputer(self.reward_config)
        self.state = AgentState()

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self.state = AgentState()
        self.reward_computer.reset()

    def _valid_actions_for_game(self, game: str, observation: str = "") -> List[str]:
        """Return list of valid action names for prompt (for take_action)."""
        if game == GAME_OVERCOOKED:
            return list(OVERCOOKED_VALID_ACTIONS)
        if game == GAME_VIDEOGAMEBENCH:
            return list(VIDEOGAMEBENCH_VALID_ACTIONS)
        if game == GAME_VIDEOGAMEBENCH_DOS:
            return list(VIDEOGAMEBENCH_DOS_VALID_KEYS)
        if game == GAME_GAMINGAGENT:
            m = re.search(r"[Vv]alid\s+actions?\s*[:\-]\s*(.+?)(?:\n|\.|$)", observation or "")
            if m:
                raw = m.group(1).strip()
                return [a.strip() for a in re.split(r"[,;]", raw) if a.strip()]
            return ["no-op", "stay"]
        return ["stay", "no-op"]

    def _format_active_skill(self) -> str:
        """Format the active skill guidance for the prompt."""
        s = self.state
        if not s.active_skill_id or not s.active_skill_guidance:
            return "(no active skill — use select_skill to choose one)"

        g = s.active_skill_guidance
        parts = [f"** Active skill: {g.get('skill_name', s.active_skill_id)} **"]

        if g.get("why_selected"):
            parts.append(f"  Why: {g['why_selected'][:120]}")

        if g.get("execution_hint"):
            parts.append(f"  How: {g['execution_hint'][:120]}")

        protocol = g.get("protocol", {})
        steps = protocol.get("steps", [])
        if steps:
            parts.append(f"  Plan ({s.skill_step_index}/{len(steps)} done):")
            for i, step in enumerate(steps[:7]):
                marker = "→" if i == s.skill_step_index else " "
                done_mark = "✓" if i < s.skill_step_index else " "
                parts.append(f"    {done_mark}{marker} {i+1}. {step}")

        if g.get("preconditions"):
            parts.append(f"  Preconditions: {'; '.join(g['preconditions'][:3])}")
        if g.get("termination_hint"):
            parts.append(f"  Done when: {g['termination_hint'][:100]}")
        if g.get("failure_modes"):
            parts.append(f"  Watch for: {'; '.join(g['failure_modes'][:2])}")

        return "\n".join(parts)

    def _build_prompt(
        self,
        observation: str,
        info: Optional[Dict[str, Any]],
        last_tool_name: Optional[str],
        last_tool_result: Optional[str],
    ) -> str:
        game = self.game_hint or info.get("game") or detect_game(observation)
        valid_actions = self._valid_actions_for_game(game, observation)
        skill_text = skill_bank_to_text(self.skill_bank)
        s = self.state

        can_select = s.steps_since_retrieval >= self.retrieval_budget_n or s.stuck_counter >= 3 or s.active_skill_id is None

        prompt = (
            "You are a VLM decision-making agent. You make decisions by selecting and following skills.\n\n"
            "Tools:\n"
            "- select_skill: choose a skill to execute given current state. "
            "Returns the full plan (protocol steps, preconditions, when done, what can fail). "
            "Args: {\"key\": \"<what you want to achieve>\"}\n"
            "- take_action: execute one environment action. "
            "Args: {\"action\": \"<valid_action>\"}\n"
            "- reward: compute reward after action (called by system). Args: {}\n"
            "- get_state_summary: (called by system) key=value state summary.\n"
            "- get_intention: (called by system) [TAG] subgoal phrase.\n\n"
            "Decision loop:\n"
            "  1) State summary is computed (you see it below).\n"
            "  2) If no active skill or current skill failed/completed → select_skill to choose a new one.\n"
            "  3) Follow the active skill's protocol steps via take_action.\n"
            "  4) Check termination: if done-condition met → select next skill. If failure detected → select different skill.\n\n"
            "Key rules:\n"
            "- select_skill is the ONLY way to query skills, rollouts, and prior experience.\n"
            "- Always have an active skill. If stuck or the skill isn't working, select a new one.\n"
            "- Follow the protocol steps in order. The plan tells you what to do.\n"
        )

        if not can_select:
            prompt += f"- Skill selection budget: wait {self.retrieval_budget_n - s.steps_since_retrieval} more steps (or get stuck) to re-select.\n"
        else:
            prompt += "- You CAN select a new skill this step.\n"

        prompt += "\nValid actions for take_action: " + ", ".join(valid_actions) + "\n\n"
        prompt += "Available skills:\n" + skill_text + "\n\n"

        prompt += "Current observation:\n" + (observation[:3000] if observation else "(none)") + "\n\n"

        prompt += "─── Active Skill ───\n"
        prompt += self._format_active_skill() + "\n\n"

        prompt += "─── Agent State ───\n"
        prompt += "- intention: " + (s.current_intention or "(none)") + "\n"
        prompt += "- progress: " + (" | ".join(s.progress_notes[-MAX_PROGRESS_NOTES:]) or "(none)") + "\n"
        prompt += "- last_actions: " + (", ".join(str(a) for a in s.last_actions[-MAX_LAST_ACTIONS:]) or "(none)") + "\n"
        prompt += "- stuck_counter: " + str(s.stuck_counter) + "\n"
        if s.last_reward:
            prompt += "- last_reward: " + repr(s.last_reward) + "\n"
        if s.last_state_summary:
            prompt += "- state_summary: " + s.last_state_summary[:400] + "\n"

        if last_tool_name:
            prompt += "\nLast tool: " + last_tool_name + "\n"
            if last_tool_result:
                prompt += "Result: " + (str(last_tool_result)[:800]) + "\n"

        prompt += (
            "\nOutput format (strict):\n"
            "THOUGHT: <at most 2 sentences>\nTOOL: <tool_name>\nARGS: <json object>\n"
            "Example: TOOL: select_skill\nARGS: {\"key\": \"navigate to pot with onion\"}\n"
            "Example: TOOL: take_action\nARGS: {\"action\": \"north\"}\n"
        )
        return prompt

    def _parse_tool_response(self, reply: str, observation: str, game: str) -> Dict[str, Any]:
        """Parse LLM reply into {tool, args}. Fallback to take_action."""
        tool = TOOL_TAKE_ACTION
        args: Dict[str, Any] = {}

        tool_m = re.search(r"TOOL\s*:\s*(\w+)", reply, re.IGNORECASE)
        args_m = re.search(r"ARGS\s*:\s*(\{[\s\S]*?\})(?=\s*(?:TOOL|THOUGHT|$))", reply, re.IGNORECASE)
        if tool_m:
            raw_tool = tool_m.group(1).strip().lower()
            for t in TOOLS:
                if t == raw_tool or raw_tool in t:
                    tool = t
                    break

        if args_m:
            try:
                args = json.loads(args_m.group(1))
            except json.JSONDecodeError:
                pass

        if tool == TOOL_TAKE_ACTION and "action" not in args:
            action = extract_action(reply, game, observation)
            if action is None:
                action = _default_action(game)
            args["action"] = action

        return {"tool": tool, "args": args or {}}

    def step(
        self,
        observation: str,
        info: Optional[Dict[str, Any]] = None,
        last_tool_name: Optional[str] = None,
        last_tool_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Decide the next tool call. Returns {"tool": str, "args": dict}.
        """
        info = info or {}
        game = self.game_hint or info.get("game") or detect_game(observation or "")

        s = self.state
        if s.active_skill_plan and s.skill_step_index < len(s.active_skill_plan):
            step_spec = s.active_skill_plan[s.skill_step_index]
            action = step_spec.get("action")
            if action is not None:
                s.skill_step_index += 1
                return {"tool": TOOL_TAKE_ACTION, "args": {"action": action}}

        if ask_model is None:
            action = _default_action(game)
            return {"tool": TOOL_TAKE_ACTION, "args": {"action": action}}

        prompt = self._build_prompt(
            observation,
            info,
            last_tool_name,
            str(last_tool_result) if last_tool_result is not None else None,
        )
        reply = ask_model(prompt, model=self.model, temperature=0.3, max_tokens=400)
        out = self._parse_tool_response(reply or "", observation, game)
        return out

    def update_from_tool_result(
        self,
        tool_name: str,
        tool_result: Any,
        observation: Optional[str] = None,
        game: Optional[str] = None,
    ) -> None:
        """Update internal state after the runner executed a tool."""
        s = self.state
        s.steps_since_retrieval += 1

        if tool_name == TOOL_GET_STATE_SUMMARY:
            s.last_state_summary = str(tool_result)[:HARD_SUMMARY_CHAR_LIMIT]

        elif tool_name == TOOL_GET_INTENTION:
            s.current_intention = str(tool_result)[:200] if tool_result else ""

        elif tool_name == TOOL_TAKE_ACTION:
            s.last_actions.append(tool_result if isinstance(tool_result, (str, list)) else str(tool_result))
            if len(s.last_actions) > MAX_LAST_ACTIONS:
                s.last_actions = s.last_actions[-MAX_LAST_ACTIONS:]
            s.steps_without_progress += 1
            if isinstance(tool_result, dict) and tool_result.get("progress"):
                s.stuck_counter = 0
                s.steps_without_progress = 0
                s.progress_notes.append(tool_result.get("progress", "Progress")[:100])
            if len(s.progress_notes) > MAX_PROGRESS_NOTES:
                s.progress_notes = s.progress_notes[-MAX_PROGRESS_NOTES:]

        elif tool_name == TOOL_SELECT_SKILL:
            s.steps_since_retrieval = 0
            s.steps_without_progress = 0
            if isinstance(tool_result, dict):
                s.active_skill_id = tool_result.get("skill_id")
                s.active_skill_guidance = tool_result

                protocol = tool_result.get("protocol", {})
                steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
                if steps:
                    s.active_skill_plan = [{"action": step} for step in steps[:7]]
                else:
                    micro_plan = tool_result.get("micro_plan", [])
                    if micro_plan:
                        s.active_skill_plan = micro_plan
                    else:
                        s.active_skill_plan = None
                s.skill_step_index = 0

        elif tool_name == TOOL_REWARD:
            if isinstance(tool_result, RewardResult):
                s.last_reward = tool_result

        if tool_name == TOOL_TAKE_ACTION and s.steps_without_progress >= self.skill_abort_k:
            s.stuck_counter += 1
            s.active_skill_plan = None
            s.active_skill_id = None
            s.active_skill_guidance = None
            s.skill_step_index = 0


# ---------------------------------------------------------------------------
# Tool execution (for use by runner)
# ---------------------------------------------------------------------------

def run_tool(
    tool_name: str,
    args: Dict[str, Any],
    agent: VLMDecisionAgent,
    observation: str,
    info: Optional[Dict[str, Any]] = None,
    env: Any = None,
) -> Any:
    """
    Execute one tool and return the result.
    - take_action: env.step(args["action"])
    - get_state_summary: compact key=value state summary
    - get_intention: [TAG] subgoal phrase
    - select_skill: state-aware skill selection with full guidance package
    - reward: compute reward signals
    """
    info = info or {}
    game = agent.game_hint or info.get("game") or detect_game(observation or "")

    if tool_name == TOOL_TAKE_ACTION:
        action = args.get("action")
        if env is not None and action is not None:
            result = env.step(action)
            return result
        return action

    if tool_name == TOOL_GET_STATE_SUMMARY:
        structured = info.get("structured_state") if info else None
        return helper_get_state_summary(
            observation,
            structured_state=structured,
            game=game,
            model=agent.model,
        )

    if tool_name == TOOL_GET_INTENTION:
        summary = agent.state.last_state_summary or observation
        return helper_infer_intention(
            summary,
            game=game,
            model=agent.model,
            context={
                "last_actions": agent.state.last_actions,
                "progress_notes": agent.state.progress_notes,
                "task": info.get("task", ""),
            },
        )

    if tool_name == TOOL_SELECT_SKILL:
        key = args.get("key", "")
        current_state = _extract_current_predicates(
            agent.state.last_state_summary, observation
        )
        result = select_skill_from_bank(
            agent.skill_bank,
            key,
            current_state=current_state,
            memory=agent.memory,
            top_k=1,
        )
        return result

    if tool_name == TOOL_REWARD:
        r_env = args.get("r_env", 0.0)
        action_type = args.get("action_type", "primitive")
        skill_id = agent.state.active_skill_id
        contract = None
        if skill_id and agent.skill_bank:
            try:
                contract = agent.skill_bank.get_contract(skill_id)
            except Exception:
                pass
        return agent.reward_computer.compute_reward(
            r_env=r_env,
            action_type=action_type,
            observation=observation,
            active_skill_id=skill_id,
            skill_contract=contract,
        )

    return None


def _extract_current_predicates(
    state_summary: str,
    observation: str,
) -> Optional[Dict[str, float]]:
    """Best-effort extraction of predicate-like state from the summary.

    Parses ``key=value`` pairs from the state summary and converts boolean-ish
    values to floats for the skill selection engine.
    """
    text = state_summary or observation or ""
    if not text:
        return None

    predicates: Dict[str, float] = {}
    for part in re.split(r"\s*\|\s*", text):
        m = re.match(r"(\w+)\s*=\s*(.+)", part.strip())
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().lower()
        if val in ("true", "yes", "1"):
            predicates[key] = 1.0
        elif val in ("false", "no", "0", "none", ""):
            predicates[key] = 0.0
        else:
            try:
                predicates[key] = float(val)
            except ValueError:
                predicates[key] = 1.0

    return predicates if predicates else None


# ---------------------------------------------------------------------------
# Episode runner: run one episode with the VLM agent
# ---------------------------------------------------------------------------

def run_episode_vlm_agent(
    env: Any,
    agent: Optional[VLMDecisionAgent] = None,
    *,
    model: Optional[str] = None,
    skill_bank: Any = None,
    memory: Optional[EpisodicMemoryStore] = None,
    reward_config: Optional[RewardConfig] = None,
    task: str = "",
    max_steps: int = 1000,
    verbose: bool = False,
) -> Episode:
    """
    Run one episode with the per-step loop:
      1) get_state_summary  (system, required)
      2) select_skill       (agent chooses skill → gets protocol + guidance)
      3) take_action         (follows protocol steps or acts on guidance)
      4) get_intention       (system, required — reflects what happened)
      5) reward              (system, required)

    select_skill is the ONLY protocol to query prior rollouts and experience.
    The agent gets the skill plan from the protocol.
    """
    if agent is None:
        agent = VLMDecisionAgent(
            model=model, skill_bank=skill_bank, memory=memory,
            reward_config=reward_config,
        )
    agent.reset()

    obs, info = env.reset()
    is_multi = isinstance(obs, dict)
    if is_multi:
        obs = list(obs.values())[0] if obs else ""
    observation = str(obs) if obs else ""
    info = dict(info or {})
    info.setdefault("game", detect_game(observation))

    episode_task = task or info.get("task", "")

    last_tool_name: Optional[str] = None
    last_tool_result: Any = None
    step_count = 0
    done = False
    experiences: List[Experience] = []

    while step_count < max_steps:
        # ── Phase A: get_state_summary (system, required) ──
        summary_result = run_tool(
            TOOL_GET_STATE_SUMMARY, {}, agent, observation, info, env=None
        )
        agent.update_from_tool_result(
            TOOL_GET_STATE_SUMMARY, summary_result, observation, info.get("game")
        )
        last_tool_name = TOOL_GET_STATE_SUMMARY
        last_tool_result = summary_result

        # ── Phase B: agent decides (select_skill or take_action) ──
        out = agent.step(observation, info, last_tool_name, last_tool_result)
        tool_name = out.get("tool", TOOL_TAKE_ACTION)
        tool_args = out.get("args") or {}

        if tool_name == TOOL_SELECT_SKILL:
            result = run_tool(tool_name, tool_args, agent, observation, info, env=None)
            agent.update_from_tool_result(tool_name, result, observation, info.get("game"))
            last_tool_name = tool_name
            last_tool_result = result
            # After selection, agent decides next action
            out = agent.step(observation, info, last_tool_name, last_tool_result)
            tool_name = out.get("tool", TOOL_TAKE_ACTION)
            tool_args = out.get("args") or {}

        if tool_name != TOOL_TAKE_ACTION:
            tool_name = TOOL_TAKE_ACTION
            tool_args = {"action": _default_action(info.get("game") or detect_game(observation))}

        # Snapshot agent state BEFORE the action
        pre_action_summary = agent.state.last_state_summary or ""
        pre_action_intention = agent.state.current_intention or ""
        pre_action_skill = agent.state.active_skill_id

        # ── Phase C: take_action ──
        action = tool_args.get("action")
        if action is None:
            action = _default_action(info.get("game") or detect_game(observation))

        action_type = "CALL_SKILL" if pre_action_skill else "primitive"

        if is_multi:
            active = info.get("active_players", [])
            actions = {pid: action for pid in active} if active else {0: action}
        else:
            actions = action

        next_obs, reward, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated

        progress = None
        if reward and float(reward) != 0:
            progress = "reward " + str(reward)
        agent.update_from_tool_result(
            TOOL_TAKE_ACTION,
            {"action": action, "progress": progress} if progress else action,
            observation,
            info.get("game"),
        )

        if is_multi:
            next_observation = str(next_obs.get(list(next_obs.keys())[0], "")) if next_obs else ""
        else:
            next_observation = str(next_obs) if next_obs else ""

        # ── Phase D: update intention (system, required) ──
        intention_result = run_tool(
            TOOL_GET_INTENTION, {}, agent, next_observation, dict(next_info or {}), env=None
        )
        agent.update_from_tool_result(
            TOOL_GET_INTENTION, intention_result, next_observation, info.get("game")
        )

        # ── Phase E: reward (system, required) ──
        env_reward = float(reward) if not isinstance(reward, dict) else sum(reward.values())
        rr = run_tool(
            TOOL_REWARD,
            {"r_env": env_reward, "action_type": action_type},
            agent,
            observation,
            info,
            env=None,
        )
        agent.update_from_tool_result(TOOL_REWARD, rr, observation, info.get("game"))
        rr_dict = rr.to_dict() if isinstance(rr, RewardResult) else {"r_env": env_reward}

        # ── Build Experience ──
        exp = Experience(
            state=observation,
            action=str(action),
            reward=env_reward,
            next_state=next_observation,
            done=done,
            intentions=pre_action_intention if pre_action_intention else None,
            tasks=episode_task if episode_task else None,
            sub_tasks=pre_action_skill,
        )
        exp.idx = step_count
        exp.summary_state = pre_action_summary if pre_action_summary else None
        exp.reward_details = rr_dict
        exp.action_type = action_type
        experiences.append(exp)

        if verbose:
            print(f"  step {step_count}: action={action}  skill={pre_action_skill}  {rr}")

        last_tool_name = TOOL_REWARD
        last_tool_result = rr
        observation = next_observation
        info = dict(next_info or {})
        info.setdefault("game", detect_game(observation))
        step_count += 1

        if done:
            break

    cumulative = agent.reward_computer.cumulative
    env_name = info.get("env_name") or info.get("game") or detect_game(observation)
    game_name = info.get("game_name") or info.get("structured_state", {}).get("game") or env_name

    episode = Episode(
        experiences=experiences,
        task=episode_task or "Unspecified task",
        env_name=env_name,
        game_name=game_name,
        metadata={
            "done": done,
            "steps": step_count,
            "agent_state": {
                "current_intention": agent.state.current_intention,
                "progress_notes": agent.state.progress_notes,
                "last_actions": agent.state.last_actions,
                "active_skill_id": agent.state.active_skill_id,
            },
            "cumulative_reward": cumulative.to_dict(),
        },
    )
    episode.set_outcome()
    return episode


# LLM-first aliases (class name VLMDecisionAgent kept for backward compatibility)
LLMDecisionAgent = VLMDecisionAgent
run_episode_llm_agent = run_episode_vlm_agent
