"""
Avalon game env wrapper: state <-> natural language, gym-style multi-agent interface.

Wraps the AvalonGameEnvironment engine into a synchronous reset()/step() loop where
every agent receives a natural-language observation and returns a natural-language action.

Supports multi-agent (all players controlled externally) and single-agent (one player
controlled, the rest use a partner_policy callback).

Usage (multi-agent, from codebase root):

    from env_wrappers.avalon_nl_wrapper import AvalonNLWrapper

    env = AvalonNLWrapper(num_players=5)
    obs, info = env.reset()
    # obs: dict {player_id: str} with NL state for each player
    # info["phase"], info["roles"], info["leader"], ...

    # Each step: pass actions for the active players in the current phase
    actions = {0: "approve", 1: "reject", 2: "approve", 3: "reject", 4: "approve"}
    obs, rewards, terminated, truncated, info = env.step(actions)

Usage (single-agent):

    env = AvalonNLWrapper(num_players=5, controlled_player=0)
    obs, info = env.reset()          # obs: str (NL for player 0)
    obs, reward, term, trunc, info = env.step("approve")
"""

import itertools
import logging
import re
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

_log = logging.getLogger(__name__)

try:
    from games.games.avalon.engine import AvalonBasicConfig, AvalonGameEnvironment
except ImportError:
    AvalonBasicConfig = None  # type: ignore
    AvalonGameEnvironment = None  # type: ignore

# ---------------------------------------------------------------------------
# Phase names (human-readable)
# ---------------------------------------------------------------------------
_PHASE_NAMES = {0: "Team Selection", 1: "Team Voting", 2: "Quest Voting", 3: "Assassination"}

_ROLE_SIDE = {
    "Merlin": "Good", "Percival": "Good", "Servant": "Good",
    "Morgana": "Evil", "Mordred": "Evil", "Oberon": "Evil",
    "Minion": "Evil", "Assassin": "Evil",
}

# ---------------------------------------------------------------------------
# Compact structured state summary (for agent context / retrieval)
# ---------------------------------------------------------------------------

_PHASE_SHORT = {0: "team_select", 1: "team_vote", 2: "quest_vote", 3: "assassination"}

_OBJECTIVE_BY_PHASE = {
    0: "propose_team",
    1: "approve_or_reject_team",
    2: "pass_or_fail_quest",
    3: "choose_assassination_target",
}


def build_structured_state_summary(
    env: "AvalonGameEnvironment",
    roles: list,
    player_id: int,
) -> dict:
    """Build a compact structured dict for the Avalon game state.

    Designed to be fed into ``compact_structured_state()`` from
    ``decision_agents.agent_helper``.

    Returns:
        Dict with short key=value-friendly fields.  Example::

            {"game": "avalon", "phase": "team_vote",
             "self": "role:Percival", "progress": "quest:1/5 fail:0",
             "critical": "leader:p3 team:3",
             "objective": "approve_or_reject_team"}
    """
    phase_id = env.phase
    phase_short = _PHASE_SHORT.get(phase_id, f"p{phase_id}")

    role_id, role_name, is_good = roles[player_id]
    side = "G" if is_good else "E"

    quest_done = len(env.quest_results)
    good_wins = sum(env.quest_results) if env.quest_results else 0
    evil_wins = quest_done - good_wins

    leader = env.quest_leader
    team_size = (
        env.num_players_for_quest[env.turn]
        if env.turn < len(env.num_players_for_quest)
        else 0
    )

    summary: dict = {
        "game": "avalon",
        "phase": phase_short,
        "self": f"role:{role_name}({side})",
        "progress": f"quest:{quest_done}/5 good:{good_wins} evil:{evil_wins}",
        "critical": f"leader:p{leader} team_sz:{team_size} rd:{env.round + 1}/5",
        "objective": _OBJECTIVE_BY_PHASE.get(phase_id, ""),
    }

    # Add quest team if visible
    if env.quest_team:
        summary["critical"] += f" team:{list(env.quest_team)}"

    return summary


# ---------------------------------------------------------------------------
# NL -> action parsers
# ---------------------------------------------------------------------------
_APPROVE_WORDS = {"approve", "yes", "accept", "aye", "agree", "yea", "pass", "support", "1"}
_REJECT_WORDS = {"reject", "no", "deny", "nay", "disagree", "oppose", "fail", "0"}


def parse_vote(text: Union[str, int]) -> int:
    """Parse an approve/reject or pass/fail vote from NL text. Returns 1 or 0."""
    if isinstance(text, int):
        return 1 if text else 0
    s = str(text).strip().lower()
    for w in re.findall(r"[a-z0-9]+", s):
        if w in _APPROVE_WORDS:
            return 1
        if w in _REJECT_WORDS:
            return 0
    # Default: approve
    return 1


def parse_team(text: Union[str, List[int]], num_players: int, team_size: int) -> List[int]:
    """
    Parse a team proposal from NL text or list of ints.

    Accepts formats like:
      - "Player 0, Player 2"  /  "0, 2"  /  [0, 2]  /  "players 0 and 2"
    Returns sorted list of unique player indices, clamped to team_size.
    """
    if isinstance(text, (list, tuple)):
        ids = [int(x) for x in text if 0 <= int(x) < num_players]
    else:
        ids = [int(x) for x in re.findall(r"\d+", str(text)) if int(x) < num_players]
    # Deduplicate, clamp to team_size
    ids = list(dict.fromkeys(ids))[:team_size]
    # Pad with random players if too few
    if len(ids) < team_size:
        remaining = [i for i in range(num_players) if i not in ids]
        random.shuffle(remaining)
        ids.extend(remaining[: team_size - len(ids)])
    return ids


def parse_target(text: Union[str, int], num_players: int) -> int:
    """Parse an assassination target from NL text or int. Returns player index."""
    if isinstance(text, int):
        return max(0, min(text, num_players - 1))
    nums = [int(x) for x in re.findall(r"\d+", str(text))]
    if nums:
        return max(0, min(nums[0], num_players - 1))
    return 0


# ---------------------------------------------------------------------------
# State -> NL
# ---------------------------------------------------------------------------

def _format_role_info(player_id: int, roles: list, env: "AvalonGameEnvironment") -> str:
    """Private role information for a specific player."""
    role_id, role_name, is_good = roles[player_id]
    side = "Good" if is_good else "Evil"
    lines = [f"Your role: {role_name} ({side} side)."]

    # Merlin sees evil players (except Mordred)
    if role_id == 0:  # Merlin
        evil_players = [i for i, (rid, rn, ig) in enumerate(roles) if not ig and rid != 3]
        if evil_players:
            lines.append(f"You can see that players {evil_players} are Evil (Mordred is hidden from you).")

    # Evil players see each other (except Oberon)
    if not is_good:
        other_evil = [i for i, (rid, rn, ig) in enumerate(roles)
                      if not ig and i != player_id and rid != 4]  # exclude Oberon
        if other_evil:
            lines.append(f"Your Evil teammates (visible to you): {other_evil}.")

    # Percival sees Merlin and Morgana (but can't distinguish)
    if role_id == 1:  # Percival
        candidates = [i for i, (rid, rn, ig) in enumerate(roles) if rid in (0, 2)]
        if candidates:
            lines.append(f"You can see that players {candidates} are either Merlin or Morgana (you cannot tell which).")

    return "\n".join(lines)


def state_to_natural_language(
    env: "AvalonGameEnvironment",
    roles: list,
    player_id: int,
    discussion_log: Optional[List[str]] = None,
    team_proposal: Optional[List[int]] = None,
    vote_result: Optional[dict] = None,
) -> str:
    """
    Convert Avalon game state to natural-language description for one player.

    Args:
        env: AvalonGameEnvironment instance.
        roles: List of (role_id, role_name, is_good) tuples.
        player_id: Player receiving this observation.
        discussion_log: Recent discussion messages (optional).
        team_proposal: Currently proposed team (optional).
        vote_result: Result of last vote (optional).

    Returns:
        Natural-language state description string.
    """
    phase_id = env.phase
    phase_name = _PHASE_NAMES.get(phase_id, f"Phase {phase_id}")
    num_players = len(roles)

    lines = [f"=== Avalon Game — {phase_name} ==="]
    lines.append(f"You are Player {player_id} (of {num_players} players).")
    lines.append(_format_role_info(player_id, roles, env))
    lines.append("")

    # Quest progress
    if env.quest_results:
        results_str = ", ".join("Success" if r else "Fail" for r in env.quest_results)
        good_wins = sum(env.quest_results)
        evil_wins = len(env.quest_results) - good_wins
        lines.append(f"Quest results so far: [{results_str}] (Good {good_wins} - Evil {evil_wins}).")
    else:
        lines.append("No quests completed yet.")

    lines.append(f"Current quest: {env.turn + 1} of 5.")
    lines.append(f"Current round: {env.round + 1} of 5 (if 5 rejections, team auto-passes).")
    lines.append("")

    leader = env.quest_leader
    team_size = env.num_players_for_quest[env.turn] if env.turn < len(env.num_players_for_quest) else 0

    # Phase-specific information and action prompt
    if phase_id == 0:  # Team Selection
        lines.append(f"Quest leader: Player {leader}.")
        lines.append(f"Team size required: {team_size} players.")
        if player_id == leader:
            lines.append("")
            lines.append(f"You are the quest leader. Propose a team of {team_size} players.")
            lines.append(f"Reply with {team_size} player IDs (0 to {num_players - 1}), e.g.: \"0, 2, 3\"")
        else:
            lines.append("")
            lines.append(f"Waiting for Player {leader} (quest leader) to propose a team of {team_size}.")
            lines.append("You may state your opinion or remain silent.")

    elif phase_id == 1:  # Team Voting
        current_team = list(env.quest_team) if env.quest_team else []
        lines.append(f"Proposed team: {current_team}.")
        lines.append("")
        lines.append("Vote to APPROVE or REJECT this team.")
        lines.append("Reply with: approve / reject")

    elif phase_id == 2:  # Quest Voting
        current_team = list(env.quest_team) if env.quest_team else []
        if player_id in current_team:
            lines.append(f"You are on the quest team: {current_team}.")
            lines.append("")
            lines.append("Vote for the quest outcome: PASS (success) or FAIL (sabotage).")
            lines.append("Reply with: pass / fail")
            if roles[player_id][2]:  # is_good
                lines.append("(As a Good player, you should vote pass.)")
        else:
            lines.append(f"Quest team {current_team} is voting. You are not on the team.")
            lines.append("Waiting for quest result.")

    elif phase_id == 3:  # Assassination
        assassin_id = int(env.get_assassin())
        if player_id == assassin_id:
            lines.append("Good has won 3 quests, but you are the Assassin!")
            lines.append(f"Choose a player to assassinate (0 to {num_players - 1}). If you pick Merlin, Evil wins!")
            lines.append("Reply with a player ID, e.g.: \"2\"")
        else:
            lines.append("Good has won 3 quests. The Assassin is choosing a target...")
            lines.append("Waiting for assassination result.")

    # Recent discussion / vote results
    if discussion_log:
        lines.append("")
        lines.append("--- Recent Messages ---")
        for msg in discussion_log[-10:]:
            lines.append(msg)

    if vote_result:
        lines.append("")
        lines.append(f"Last vote result: {vote_result.get('outcome', 'unknown')}.")
        if "votes" in vote_result:
            lines.append(f"Votes: {vote_result['votes']}.")

    return "\n".join(lines)


def state_to_natural_language_for_all(
    env: "AvalonGameEnvironment",
    roles: list,
    **kwargs,
) -> Dict[int, str]:
    """Return NL observation for every player."""
    return {
        i: state_to_natural_language(env, roles, player_id=i, **kwargs)
        for i in range(len(roles))
    }


# ---------------------------------------------------------------------------
# Default partner policy (random)
# ---------------------------------------------------------------------------

def _random_partner_action(env, roles, player_id, phase_id):
    """Random action for non-controlled players."""
    if phase_id == 0:  # team selection (only leader acts)
        team_size = env.num_players_for_quest[env.turn]
        return random.sample(range(len(roles)), team_size)
    elif phase_id == 1:
        return random.choice([0, 1])
    elif phase_id == 2:
        # Good players always pass, evil players randomly fail
        return 1 if roles[player_id][2] else random.choice([0, 1])
    elif phase_id == 3:
        return random.randint(0, len(roles) - 1)
    return 0


def _retrieve_skill_hint(skill_bank: Any, obs_text: str, game: str = "avalon") -> str:
    """Fast CPU-only top-1 skill retrieval.  Returns a short hint or ''."""
    if skill_bank is None:
        return ""
    try:
        has_items = (
            (hasattr(skill_bank, "__len__") and len(skill_bank) > 0)
            or (hasattr(skill_bank, "skill_ids")
                and len(list(skill_bank.skill_ids)) > 0)
        )
        if not has_items:
            return ""
        from scripts.qwen3_decision_agent import get_top_k_skill_candidates
        candidates = get_top_k_skill_candidates(
            skill_bank, obs_text, game_name=game, top_k=1,
        )
        if candidates:
            name = candidates[0].get("skill_name", "")
            hint = candidates[0].get("execution_hint", "")
            if name or hint:
                return f"Strategy hint: {name}. {hint}".strip()[:200]
    except Exception:
        pass
    return ""


def _make_llm_partner_policy(
    base_urls: List[str],
    model_name: str,
    adapter_name: str = "action_taking",
    skill_bank: Any = None,
    game_name: str = "avalon",
) -> Callable:
    """Create an LLM-based partner policy that queries vLLM for actions.

    Returns a callable with the same signature as ``_random_partner_action``.
    Uses round-robin across *base_urls* for load balancing.
    Opponents use the ``action_taking`` LoRA so they improve alongside the
    controlled agent (self-play dynamics) without the skill-selection overhead.
    When a *skill_bank* is provided the top-1 skill hint is injected into the
    prompt (CPU-only retrieval, zero extra LLM calls).
    Falls back to random on any failure.
    """
    import requests as _requests

    _url_cycle = itertools.cycle(base_urls)

    def _llm_partner_action(env, roles, player_id, phase_id):
        try:
            obs = state_to_natural_language(env, roles, player_id=player_id)
            obs_short = obs[:1500]
            phase_name = _PHASE_NAMES.get(phase_id, f"Phase {phase_id}")
            num_players = len(roles)

            skill_line = _retrieve_skill_hint(skill_bank, obs_short, game_name)
            skill_prefix = f"{skill_line}\n" if skill_line else ""

            if phase_id == 0:
                team_size = env.num_players_for_quest[env.turn]
                prompt = (
                    f"{obs_short}\n\n{skill_prefix}"
                    f"Pick {team_size} players for the quest team. "
                    f"Reply with just the player numbers separated by commas "
                    f"(e.g. \"0,2,3\").\nTeam:"
                )
            elif phase_id == 1:
                prompt = (
                    f"{obs_short}\n\n{skill_prefix}"
                    f"Vote: approve or reject this team. "
                    f"Reply with just one word.\nVote:"
                )
            elif phase_id == 2:
                prompt = (
                    f"{obs_short}\n\n{skill_prefix}"
                    f"Quest vote: pass or fail. "
                    f"Reply with just one word.\nVote:"
                )
            elif phase_id == 3:
                prompt = (
                    f"{obs_short}\n\n{skill_prefix}"
                    f"Choose a player to assassinate (0 to {num_players - 1}). "
                    f"Reply with just the player number.\nTarget:"
                )
            else:
                return _random_partner_action(env, roles, player_id, phase_id)

            url = next(_url_cycle)
            resp = _requests.post(
                f"{url}/completions",
                json={
                    "model": adapter_name,
                    "prompt": prompt,
                    "max_tokens": 16,
                    "temperature": 0.4,
                    "stop": ["\n"],
                },
                timeout=15,
            )
            text = resp.json()["choices"][0]["text"].strip()

            if phase_id == 0:
                team_size = env.num_players_for_quest[env.turn]
                return parse_team(text, num_players, team_size)
            elif phase_id in (1, 2):
                return parse_vote(text)
            elif phase_id == 3:
                return parse_target(text, num_players)

        except Exception as exc:
            _log.debug("LLM partner action failed for player %d: %s", player_id, exc)
            return _random_partner_action(env, roles, player_id, phase_id)

    return _llm_partner_action


def _make_api_partner_policy(
    api_model: str,
    api_base: str = "https://openrouter.ai/api/v1",
    api_key: Optional[str] = None,
) -> Callable:
    """Create an API-based partner policy for external opponent models.

    Calls an external chat API (e.g. OpenRouter) instead of local vLLM,
    enabling training against a fixed strong opponent to break self-play
    reward inflation.  Falls back to random on any failure.
    """
    import openai as _openai

    if api_key is None:
        try:
            from api_keys import open_router_api_key
            api_key = open_router_api_key
        except ImportError:
            import os
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

    _client = _openai.OpenAI(base_url=api_base, api_key=api_key, max_retries=2)

    def _api_partner_action(env, roles, player_id, phase_id):
        try:
            obs = state_to_natural_language(env, roles, player_id=player_id)
            obs_short = obs[:2000]
            num_players = len(roles)

            if phase_id == 0:
                team_size = env.num_players_for_quest[env.turn]
                prompt = (
                    f"{obs_short}\n\n"
                    f"Pick {team_size} players for the quest team. "
                    f"Reply with just the player numbers separated by commas "
                    f"(e.g. \"0,2,3\").\nTeam:"
                )
            elif phase_id == 1:
                prompt = (
                    f"{obs_short}\n\n"
                    f"Vote: approve or reject this team. "
                    f"Reply with just one word.\nVote:"
                )
            elif phase_id == 2:
                prompt = (
                    f"{obs_short}\n\n"
                    f"Quest vote: pass or fail. "
                    f"Reply with just one word.\nVote:"
                )
            elif phase_id == 3:
                prompt = (
                    f"{obs_short}\n\n"
                    f"Choose a player to assassinate (0 to {num_players - 1}). "
                    f"Reply with just the player number.\nTarget:"
                )
            else:
                return _random_partner_action(env, roles, player_id, phase_id)

            resp = _client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=32,
            )
            text = (resp.choices[0].message.content or "").strip()

            if phase_id == 0:
                team_size = env.num_players_for_quest[env.turn]
                return parse_team(text, num_players, team_size)
            elif phase_id in (1, 2):
                return parse_vote(text)
            elif phase_id == 3:
                return parse_target(text, num_players)

        except Exception as exc:
            _log.debug("API partner action failed for player %d: %s", player_id, exc)
            return _random_partner_action(env, roles, player_id, phase_id)

    return _api_partner_action


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class AvalonNLWrapper:
    """
    Gym-style wrapper for the Avalon game engine with natural-language observations and actions.

    Multi-agent mode (default):
        - reset() -> (obs_dict, info)   where obs_dict = {player_id: nl_string}
        - step(actions) -> (obs_dict, rewards_dict, terminated, truncated, info)
          actions: {player_id: nl_action_string} for active players in current phase

    Single-agent mode (controlled_player is set):
        - reset() -> (obs_str, info)
        - step(action_str) -> (obs_str, reward, terminated, truncated, info)
          Other players use partner_policy.

    The wrapper advances through game phases one at a time. Each call to step()
    processes one phase transition. The game ends when env.done is True.
    """

    def __init__(
        self,
        num_players: int = 5,
        controlled_player: Optional[int] = None,
        partner_policy: Optional[Callable] = None,
        merlin: bool = True,
        percival: bool = False,
        morgana: bool = False,
        mordred: bool = False,
        oberon: bool = False,
        seed: Optional[int] = None,
        vllm_base_urls: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        skill_bank: Any = None,
        opponent_model: Optional[str] = None,
        opponent_api_base: Optional[str] = None,
    ):
        """
        Args:
            num_players: Number of players (5-10).
            controlled_player: If set, single-agent mode controlling this player.
                If None, multi-agent mode (all players controlled externally).
            partner_policy: Callable(env, roles, player_id, phase_id) -> action
                for non-controlled players in single-agent mode.
                If None and opponent_model is set, uses API policy.
                If None and vllm_base_urls is set, uses LLM policy.
                If None and no URLs, uses random policy.
            merlin..oberon: Role flags passed to AvalonBasicConfig.
            seed: Optional random seed for reproducibility.
            vllm_base_urls: vLLM server URLs for LLM-based partner policy.
            model_name: Model name for vLLM requests.
            skill_bank: Per-game skill bank for opponent skill hints.
            opponent_model: External API model name for opponents (e.g.
                "gpt-5-mini").  Takes priority over vLLM self-play.
            opponent_api_base: API base URL (default: OpenRouter).
        """
        if AvalonBasicConfig is None:
            raise ImportError(
                "Cannot import AvalonGameEnvironment. "
                "Make sure games/ is on your Python path and dependencies are installed."
            )
        self._num_players = num_players
        self._controlled_player = controlled_player
        self._multi_agent = controlled_player is None
        if partner_policy is not None:
            self._partner_policy = partner_policy
        elif opponent_model:
            self._partner_policy = _make_api_partner_policy(
                opponent_model,
                api_base=opponent_api_base or "https://openrouter.ai/api/v1",
            )
        elif vllm_base_urls and model_name:
            self._partner_policy = _make_llm_partner_policy(
                vllm_base_urls, model_name,
                skill_bank=skill_bank, game_name="avalon",
            )
        else:
            self._partner_policy = _random_partner_action
        self._role_flags = dict(
            merlin=merlin, percival=percival,
            morgana=morgana, mordred=mordred, oberon=oberon,
        )
        self._seed = seed

        self.env: Optional[AvalonGameEnvironment] = None
        self.roles: list = []
        self._discussion_log: List[str] = []
        self._last_vote_result: Optional[dict] = None

    # ---- Gym-like interface ----

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Any, dict]:
        """
        Reset the Avalon game.

        Returns:
            Multi-agent:  ({player_id: nl_str, ...}, info)
            Single-agent: (nl_str, info)
        """
        if seed is not None:
            self._seed = seed
        if self._seed is not None:
            random.seed(self._seed)
            import numpy as np
            np.random.seed(self._seed)

        config = AvalonBasicConfig.from_num_players(self._num_players, **self._role_flags)
        self.env = AvalonGameEnvironment(config)
        self.roles = self.env.get_roles()
        self._discussion_log = []
        self._last_vote_result = None
        self._prev_potential = 0.0

        info = self._build_info()
        obs = self._build_obs()
        return obs, info

    def step(
        self,
        action: Union[Dict[int, Any], str, int, list],
    ) -> Tuple[Any, Any, bool, bool, dict]:
        """
        Advance one game phase with the given action(s).

        Args:
            Multi-agent:  action = {player_id: action_value} for active players.
            Single-agent: action = str or int for the controlled player.

        Returns:
            Multi-agent:  (obs_dict, rewards_dict, terminated, truncated, info)
            Single-agent: (obs_str, reward_float, terminated, truncated, info)
        """
        if self.env is None or self.env.done:
            raise RuntimeError("Game is not running. Call reset() first.")

        phase_id = self.env.phase

        if self._multi_agent:
            actions = action if isinstance(action, dict) else {}
            self._process_phase_multi(phase_id, actions)
        else:
            self._process_phase_single(phase_id, action)

        terminated = self.env.done
        truncated = False
        info = self._build_info()
        obs = self._build_obs()
        rewards = self._build_rewards() if terminated else self._zero_rewards()

        if not terminated and not self._multi_agent:
            rewards = rewards + self._shaping_reward()

        if self._multi_agent:
            return obs, rewards, terminated, truncated, info
        else:
            return obs, rewards, terminated, truncated, info

    # ---- Phase processing (multi-agent) ----

    def _process_phase_multi(self, phase_id: int, actions: Dict[int, Any]) -> None:
        """Process one phase with actions from all agents."""
        if phase_id == 0:
            leader = self.env.quest_leader
            team_size = self.env.get_team_size()
            leader_action = actions.get(leader, None)
            if leader_action is None:
                team = random.sample(range(self._num_players), team_size)
            else:
                team = parse_team(leader_action, self._num_players, team_size)
            self.env.choose_quest_team(team=frozenset(team), leader=leader)
            self._last_vote_result = None
            self._discussion_log.append(f"Player {leader} proposed team: {team}")

        elif phase_id == 1:
            votes = []
            for i in range(self._num_players):
                v = actions.get(i, 1)  # default approve
                votes.append(parse_vote(v))
            _, _, approved = self.env.gather_team_votes(votes)
            self._last_vote_result = {
                "outcome": "Approved" if approved else "Rejected",
                "votes": {i: ("approve" if v else "reject") for i, v in enumerate(votes)},
            }
            self._discussion_log.append(
                f"Team vote: {'Approved' if approved else 'Rejected'} "
                f"({sum(votes)} approve, {len(votes) - sum(votes)} reject)."
            )

        elif phase_id == 2:
            current_team = list(self.env.get_current_quest_team())
            votes = []
            for pid in current_team:
                v = actions.get(pid, 1)
                votes.append(parse_vote(v))
            _, _, succeeded, num_fails = self.env.gather_quest_votes(votes)
            self._last_vote_result = {
                "outcome": "Success" if succeeded else "Failed",
                "num_fails": num_fails,
            }
            self._discussion_log.append(
                f"Quest {'succeeded' if succeeded else 'failed'} ({num_fails} fail vote(s))."
            )

        elif phase_id == 3:
            assassin_id = int(self.env.get_assassin())
            target_action = actions.get(assassin_id, 0)
            target = parse_target(target_action, self._num_players)
            _, _, good_wins = self.env.choose_assassination_target(assassin_id, target)
            self._discussion_log.append(
                f"Assassin (Player {assassin_id}) targeted Player {target}. "
                f"{'Good wins!' if good_wins else 'Evil wins!'}"
            )

    # ---- Phase processing (single-agent) ----

    def _process_phase_single(self, phase_id: int, action: Any) -> None:
        """Process one phase with action from controlled player + partner policy."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cp = self._controlled_player
        actions: Dict[int, Any] = {}

        if phase_id == 0:
            leader = self.env.quest_leader
            if leader == cp:
                actions[leader] = action
            else:
                actions[leader] = self._partner_policy(self.env, self.roles, leader, phase_id)

        elif phase_id == 1:
            actions[cp] = action
            partner_ids = [i for i in range(self._num_players) if i != cp]
            with ThreadPoolExecutor(max_workers=len(partner_ids)) as pool:
                futures = {
                    pool.submit(self._partner_policy, self.env, self.roles, pid, phase_id): pid
                    for pid in partner_ids
                }
                for fut in as_completed(futures):
                    pid = futures[fut]
                    try:
                        actions[pid] = fut.result()
                    except Exception:
                        actions[pid] = _random_partner_action(self.env, self.roles, pid, phase_id)

        elif phase_id == 2:
            current_team = list(self.env.get_current_quest_team())
            partner_pids = [pid for pid in current_team if pid != cp]
            if cp in current_team:
                actions[cp] = action
            if partner_pids:
                with ThreadPoolExecutor(max_workers=len(partner_pids)) as pool:
                    futures = {
                        pool.submit(self._partner_policy, self.env, self.roles, pid, phase_id): pid
                        for pid in partner_pids
                    }
                    for fut in as_completed(futures):
                        pid = futures[fut]
                        try:
                            actions[pid] = fut.result()
                        except Exception:
                            actions[pid] = _random_partner_action(self.env, self.roles, pid, phase_id)

        elif phase_id == 3:
            assassin_id = int(self.env.get_assassin())
            if assassin_id == cp:
                actions[assassin_id] = action
            else:
                actions[assassin_id] = self._partner_policy(
                    self.env, self.roles, assassin_id, phase_id
                )

        self._process_phase_multi(phase_id, actions)

    # ---- Observations ----

    def _build_obs(self) -> Any:
        """Build observation(s) for the current state."""
        if self.env is None:
            return {} if self._multi_agent else ""
        if self._multi_agent:
            return state_to_natural_language_for_all(
                self.env, self.roles,
                discussion_log=self._discussion_log,
                vote_result=self._last_vote_result,
            )
        else:
            return state_to_natural_language(
                self.env, self.roles,
                player_id=self._controlled_player,
                discussion_log=self._discussion_log,
                vote_result=self._last_vote_result,
            )

    # ---- Rewards ----

    def _build_rewards(self) -> Any:
        """Rewards at game end. +1 if your side won, 0 otherwise."""
        if self._multi_agent:
            return {
                i: (1.0 if (self.roles[i][2] == self.env.good_victory) else 0.0)
                for i in range(self._num_players)
            }
        else:
            cp = self._controlled_player
            is_good = self.roles[cp][2]
            return 1.0 if (is_good == self.env.good_victory) else 0.0

    def _zero_rewards(self) -> Any:
        if self._multi_agent:
            return {i: 0.0 for i in range(self._num_players)}
        else:
            return 0.0

    def _shaping_reward(self) -> float:
        """Potential-based intermediate reward for single-agent mode.

        Tracks quest success/failure relative to the controlled player's
        alignment.  Policy-invariant (potential-based) so it doesn't
        change the optimal policy — only gives per-step signal.
        """
        if self.env is None or not self.env.quest_results:
            return 0.0
        cp = self._controlled_player
        is_good = self.roles[cp][2] if cp is not None else True
        good_wins = sum(self.env.quest_results)
        evil_wins = len(self.env.quest_results) - good_wins
        if is_good:
            potential = (good_wins - evil_wins) * 0.1
        else:
            potential = (evil_wins - good_wins) * 0.1
        prev = getattr(self, "_prev_potential", 0.0)
        self._prev_potential = potential
        return potential - prev

    # ---- Info ----

    def _build_info(self) -> dict:
        """Build info dict with game metadata."""
        if self.env is None:
            return {}
        phase_id = self.env.phase

        # Structured summary for the primary player (controlled or player 0)
        primary_pid = self._controlled_player if self._controlled_player is not None else 0
        structured = build_structured_state_summary(
            self.env, self.roles, player_id=primary_pid,
        )

        return {
            "env_name": "avalon",
            "game_name": "avalon",
            "phase": phase_id,
            "phase_name": _PHASE_NAMES.get(phase_id, f"Phase {phase_id}"),
            "turn": self.env.turn,
            "round": self.env.round,
            "leader": self.env.quest_leader,
            "team_size": (
                self.env.num_players_for_quest[self.env.turn]
                if self.env.turn < len(self.env.num_players_for_quest)
                else 0
            ),
            "quest_results": list(self.env.quest_results),
            "quest_team": list(self.env.quest_team) if self.env.quest_team else [],
            "roles": [
                {"player_id": i, "role_id": int(rid), "role_name": rn, "is_good": bool(ig)}
                for i, (rid, rn, ig) in enumerate(self.roles)
            ],
            "done": self.env.done,
            "good_victory": self.env.good_victory if self.env.done else None,
            "num_players": self._num_players,
            "discussion_log": list(self._discussion_log),
            "last_vote_result": self._last_vote_result,
            "active_players": self._get_active_players(),
            "expected_action": self._get_expected_action_description(),
            "structured_state": structured,
        }

    def _get_active_players(self) -> List[int]:
        """Return list of player IDs that need to act in the current phase."""
        if self.env is None or self.env.done:
            return []
        phase_id = self.env.phase
        if phase_id == 0:
            return [self.env.quest_leader]
        elif phase_id == 1:
            return list(range(self._num_players))
        elif phase_id == 2:
            return list(self.env.get_current_quest_team()) if self.env.quest_team else []
        elif phase_id == 3:
            return [int(self.env.get_assassin())]
        return []

    def _get_expected_action_description(self) -> str:
        """Human-readable description of what action is expected."""
        if self.env is None or self.env.done:
            return "Game over."
        phase_id = self.env.phase
        if phase_id == 0:
            ts = self.env.get_team_size()
            return f"Leader proposes a team of {ts} (list of player IDs)."
        elif phase_id == 1:
            return "All players vote: approve / reject."
        elif phase_id == 2:
            return "Quest team members vote: pass / fail."
        elif phase_id == 3:
            return "Assassin chooses a target player ID."
        return "Unknown phase."

    # ---- Properties ----

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def multi_agent(self) -> bool:
        return self._multi_agent

    @property
    def done(self) -> bool:
        return self.env.done if self.env else True

    @property
    def action_space(self):
        """Informal; actual actions vary by phase. See expected_action in info."""
        return None

    @property
    def observation_space(self):
        """Observations are NL strings; no formal gym Space."""
        return None
