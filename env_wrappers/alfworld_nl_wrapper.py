"""ALFWorld NL wrapper for the COS-PLAY co-evolution loop.

Wraps ALFWorld's TextWorld-based household environment into the standard
COS-PLAY Gymnasium-like NL interface::

    obs_nl, info = env.reset()
    obs_nl, reward, terminated, truncated, info = env.step("go to countertop 1")

``info["action_names"]`` contains the admissible commands at each step.

``info["structured_state"]`` provides predicate-checkable state keys
compatible with ``decision_agents.protocol_utils.check_predicate``.

``info["effect_tags"]`` provides per-step effect tags from the canonical
``TASK_EFFECT_SUBSET["alfworld"]`` vocabulary for the skill-selection LoRA.

ALFWorld data must be downloaded (``alfworld-download``).  The wrapper
manages its own ``textworld.gym`` environment internally, cycling through
game files on each ``reset()``.

Task types (from ALFWorld):
  1: pick_and_place_simple
  2: look_at_obj_in_light
  3: pick_clean_then_place_in_recep
  4: pick_heat_then_place_in_recep
  5: pick_cool_then_place_in_recep
  6: pick_two_obj_and_place
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ALFWorldNLWrapper:
    """COS-PLAY NL wrapper around ALFWorld (TextWorld backend).

    Parameters
    ----------
    split : str
        ``"train"``, ``"eval_in_distribution"``, or ``"eval_out_of_distribution"``.
    task_types : list[int]
        Which ALFWorld task types to include (1-6).  Default: all.
    max_steps : int
        Maximum steps per episode.
    shuffle : bool
        Whether to shuffle game files on init.
    """

    TASK_TYPES = {
        1: "pick_and_place_simple",
        2: "look_at_obj_in_light",
        3: "pick_clean_then_place_in_recep",
        4: "pick_heat_then_place_in_recep",
        5: "pick_cool_then_place_in_recep",
        6: "pick_two_obj_and_place",
    }

    def __init__(
        self,
        split: str = "train",
        task_types: Optional[List[int]] = None,
        max_steps: int = 50,
        shuffle: bool = True,
    ):
        import alfworld
        import textworld
        import textworld.gym

        self._max_steps = max_steps
        self._step_count = 0
        self._done = False
        self._goal = ""
        self._action_history: List[str] = []
        self._cumulative_reward = 0.0
        self._prev_obs = ""
        self._inventory: List[str] = []
        self._current_location = ""
        self._visited_locations: set = set()
        self._picked_up: set = set()
        self._placed: set = set()
        self._transformed: set = set()  # cleaned/heated/cooled

        data_root = alfworld.ALFWORLD_DATA
        if split == "train":
            data_path = os.path.join(data_root, "json_2.1.1", "train")
        elif split == "eval_in_distribution":
            data_path = os.path.join(data_root, "json_2.1.1", "valid_seen")
        else:
            data_path = os.path.join(data_root, "json_2.1.1", "valid_unseen")

        wanted_types = set()
        for tt in (task_types or list(self.TASK_TYPES.keys())):
            if tt in self.TASK_TYPES:
                wanted_types.add(self.TASK_TYPES[tt])

        self._game_files: List[str] = []
        for root, _dirs, files in os.walk(data_path, topdown=False):
            if "traj_data.json" not in files:
                continue
            game_file = os.path.join(root, "game.tw-pddl")
            if not os.path.exists(game_file):
                continue
            if "movable" in root or "Sliced" in root:
                continue

            with open(os.path.join(root, "traj_data.json")) as f:
                traj = json.load(f)
            if traj.get("task_type") not in wanted_types:
                continue

            with open(game_file) as f:
                gdata = json.load(f)
            if not gdata.get("solvable", False):
                continue

            self._game_files.append(game_file)

        if shuffle:
            random.shuffle(self._game_files)

        logger.info(
            "ALFWorld: %d solvable games in split=%s", len(self._game_files), split,
        )
        if not self._game_files:
            raise RuntimeError(
                f"No ALFWorld games found in {data_path}. "
                "Run `alfworld-download` first."
            )

        self._game_idx = 0
        self._tw_env: Any = None
        self._request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
        )

    def _make_tw_env(self, game_file: str) -> Any:
        """Create a fresh TextWorld gym env for one game file."""
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredInfos,
        )

        env_id = textworld.gym.register_games(
            [game_file],
            self._request_infos,
            batch_size=1,
            max_episode_steps=self._max_steps,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        env = textworld.gym.make(env_id)
        return env

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception:
                pass

        game_file = self._game_files[self._game_idx % len(self._game_files)]
        self._game_idx += 1

        self._tw_env = self._make_tw_env(game_file)
        obs_list, infos = self._tw_env.reset()

        self._step_count = 0
        self._done = False
        self._action_history = []
        self._cumulative_reward = 0.0
        self._inventory = []
        self._visited_locations = set()
        self._picked_up = set()
        self._placed = set()
        self._transformed = set()

        obs_text = obs_list[0] if isinstance(obs_list, (list, tuple)) else str(obs_list)
        self._prev_obs = obs_text
        admissible = self._get_admissible(infos)

        self._goal = self._extract_goal(obs_text)
        self._current_location = self._extract_location(obs_text)

        self._current_task_type = "unknown"
        for tt_name in self.TASK_TYPES.values():
            if tt_name in game_file:
                self._current_task_type = tt_name
                break

        from decision_agents.protocol_utils import compute_alfworld_structured_state
        structured = compute_alfworld_structured_state(
            goal=self._goal,
            location=self._current_location,
            step=0,
            action_history=[],
            inventory=[],
            cumulative_reward=0.0,
        )

        info: Dict[str, Any] = {
            "action_names": admissible,
            "env_name": "alfworld",
            "game_name": "alfworld",
            "goal": self._goal,
            "game_file": game_file,
            "structured_state": structured,
            "effect_tags": {"state_observed": "true"},
        }
        return obs_text, info

    def step(
        self, action: str,
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done or self._tw_env is None:
            return "", 0.0, True, False, {"action_names": [], "effect_tags": {}}

        self._step_count += 1
        self._action_history.append(action)

        try:
            obs_list, reward_list, done_list, infos = self._tw_env.step([action])
        except Exception as exc:
            logger.warning("ALFWorld step error: %s", exc)
            return str(exc), 0.0, True, False, {"action_names": [], "effect_tags": {}}

        obs_text = obs_list[0] if isinstance(obs_list, (list, tuple)) else str(obs_list)
        raw_reward = float(reward_list[0]) if isinstance(reward_list, (list, tuple)) else float(reward_list)
        done = bool(done_list[0]) if isinstance(done_list, (list, tuple)) else bool(done_list)

        # Dense reward shaping: small bonuses for task-relevant progress
        shaping = self._compute_shaping_reward(action, obs_text, raw_reward)
        reward = raw_reward + shaping

        self._cumulative_reward += reward

        truncated = self._step_count >= self._max_steps and not done
        terminated = done
        self._done = terminated or truncated

        admissible = self._get_admissible(infos) if not self._done else []

        # Track inventory and location heuristically
        self._update_inventory(action, obs_text)
        new_loc = self._extract_location(obs_text)
        if new_loc:
            self._current_location = new_loc

        from decision_agents.protocol_utils import (
            compute_alfworld_effects,
            compute_alfworld_structured_state,
        )
        effects = compute_alfworld_effects(
            action=action,
            prev_obs=self._prev_obs,
            curr_obs=obs_text,
            reward=reward,
        )
        structured = compute_alfworld_structured_state(
            goal=self._goal,
            location=self._current_location,
            step=self._step_count,
            action_history=self._action_history,
            inventory=self._inventory,
            cumulative_reward=self._cumulative_reward,
        )
        self._prev_obs = obs_text

        info: Dict[str, Any] = {
            "action_names": admissible,
            "env_name": "alfworld",
            "game_name": "alfworld",
            "goal": self._goal,
            "structured_state": structured,
            "effect_tags": effects,
            "raw_env_reward": raw_reward,
        }
        return obs_text, reward, terminated, truncated, info

    def close(self) -> None:
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception:
                pass
            self._tw_env = None

    def _update_inventory(self, action: str, obs: str) -> None:
        """Heuristically track inventory from action + observation text."""
        act_l = action.lower()
        if act_l.startswith("take ") or act_l.startswith("pick up "):
            parts = act_l.replace("take ", "").replace("pick up ", "").split(" from ")
            obj = parts[0].strip()
            if "you pick up" in obs.lower() or "you take" in obs.lower():
                if obj and obj not in self._inventory:
                    self._inventory.append(obj)
        elif act_l.startswith("put "):
            parts = act_l.replace("put ", "").split(" in ")
            obj = parts[0].strip()
            if obj in self._inventory:
                self._inventory.remove(obj)

    @staticmethod
    def _get_admissible(infos: Dict[str, Any]) -> List[str]:
        cmds = infos.get("admissible_commands", [[]])
        if isinstance(cmds, list) and cmds and isinstance(cmds[0], list):
            return cmds[0]
        if isinstance(cmds, list):
            return cmds
        return ["look"]

    def _compute_shaping_reward(self, action: str, obs: str, env_reward: float) -> float:
        """Dense reward shaping for GRPO: small bonuses for task-relevant progress.

        Returns a small shaping bonus (0 to 0.05 per step).  The total
        shaping across an episode is capped at ~0.3 so it never dominates
        the binary success reward (1.0).
        """
        if env_reward > 0:
            return 0.0  # task already succeeded — no shaping needed

        bonus = 0.0
        obs_low = obs.lower()
        act_low = action.lower()
        goal_low = self._goal.lower()

        # Navigating to a NEW location → small exploration bonus
        loc = self._extract_location(obs)
        if loc and loc not in self._visited_locations:
            self._visited_locations.add(loc)
            bonus += 0.01

        # Picking up an object (first time)
        if ("you pick up" in obs_low or "you take" in obs_low) and act_low not in self._picked_up:
            self._picked_up.add(act_low)
            # Bigger bonus if the picked object is mentioned in the goal
            obj_name = act_low.split("take ")[-1].split(" from ")[0].strip()
            if any(w in goal_low for w in obj_name.split() if len(w) > 2):
                bonus += 0.05
            else:
                bonus += 0.02

        # Placing/putting an object (first time)
        if "you put" in obs_low and act_low not in self._placed:
            self._placed.add(act_low)
            bonus += 0.03

        # Cleaning/heating/cooling (first time each)
        for transform in ("clean", "heat", "cool"):
            if transform in act_low and transform in goal_low:
                key = f"{transform}_{act_low}"
                if key not in self._transformed:
                    self._transformed.add(key)
                    bonus += 0.05

        # Penalty for "Nothing happens" (invalid/useless action)
        if "nothing happens" in obs_low:
            bonus -= 0.01

        return bonus

    @staticmethod
    def _extract_goal(obs: str) -> str:
        """Extract the task goal from the first observation."""
        for line in obs.split("\n"):
            line = line.strip()
            if line.startswith("Your task is to:"):
                return line.replace("Your task is to:", "").strip()
        return obs.split("\n")[0] if obs else ""

    @staticmethod
    def _extract_location(obs: str) -> str:
        """Heuristic: extract current location from observation text."""
        for line in obs.split("\n"):
            stripped = line.strip()
            low = stripped.lower()
            if "you are in" in low:
                return stripped
            if "you arrive at" in low:
                return stripped
        return ""


def make_alfworld_env(
    max_steps: int = 50,
    split: str = "train",
    task_types: Optional[List[int]] = None,
) -> ALFWorldNLWrapper:
    """Factory function matching the COS-PLAY convention."""
    return ALFWorldNLWrapper(
        split=split,
        task_types=task_types,
        max_steps=max_steps,
    )
