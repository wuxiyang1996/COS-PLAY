"""
Test Diplomacy env wrapper with dummy (LLM) agent controlling all powers.

Uses env_wrappers.DiplomacyNLWrapper and agents.dummy_agent.language_agent_action
to run phases where every power's orders are chosen by the same LLM.

Run from codebase root. The wrapper uses the local AI_Diplomacy/diplomacy package
if present. Install its dependencies so "from diplomacy import Game" works, e.g.:

  pip install coloredlogs

  Or from AI_Diplomacy:  pip install -e .

  cd /path/to/codebase
  python evaluation_evolver/test_diplomacy_dummy.py

Visualizer (watch rollout in browser):
  First start the games web server (see AgentEvolver/games/README.md):
    python AgentEvolver/games/web/server.py
  Open http://localhost:8000/diplomacy/observe in your browser.
  Then run this script with --visualizer (optionally --visualizer_url http://localhost:8000).
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Optional: add AI_Diplomacy for diplomacy package
_ai_dip = _root / "AI_Diplomacy"
if _ai_dip.exists() and str(_ai_dip) not in sys.path:
    sys.path.insert(0, str(_ai_dip))

_agentevolver = _root / "AgentEvolver"
if _agentevolver.exists() and str(_agentevolver) not in sys.path:
    sys.path.insert(0, str(_agentevolver))

from typing import Any, Dict, List, Optional, Union

import json
import urllib.request

from env_wrappers.diplomacy_nl_wrapper import DiplomacyNLWrapper
from agents.dummy_agent import (
    GAME_DIPLOMACY,
    language_agent_action,
    run_episode_with_experience_collection,
    AgentBufferManager,
)


def _diplomacy_visualizer_state(env: DiplomacyNLWrapper, info: dict, status: str = "running") -> Dict[str, Any]:
    """Build game state dict for the web visualizer (POST /api/push-state)."""
    state = {
        "game": "diplomacy",
        "status": status,
        "phase": info.get("phase", ""),
        "round": info.get("round", 0),
    }
    map_svg = env.get_map_svg()
    if map_svg:
        state["map_svg"] = map_svg
    if env.game and hasattr(env.game, "powers"):
        state["sc_counts"] = {p: len(power.centers) for p, power in env.game.powers.items()}
    return state


def _push_visualizer_state(visualizer_url: str, state: Dict[str, Any]) -> None:
    """POST state to the games web server push-state API."""
    url = f"{visualizer_url.rstrip('/')}/api/push-state"
    data = json.dumps(state).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"[visualizer] push-state returned {resp.status}")
    except Exception as e:
        print(f"[visualizer] failed to push state: {e}")


def run_diplomacy_with_dummy_agent(
    max_phases: int = 5,
    model: str = "gpt-4o-mini",
    seed: int = 42,
    verbose: bool = True,
    visualizer_url: Optional[str] = None,
    use_experience_collection: bool = False,
    experience_buffer_size: int = 10000,
    episode_buffer_size: int = 1000,
    save_episode_buffer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Diplomacy for a few phases; all powers are controlled by the dummy language agent.

    If visualizer_url is set (e.g. http://localhost:8000), state is pushed after each step
    so you can watch the rollout in the browser (open /diplomacy/observe first).

    Args:
        use_experience_collection: If True, use experience/episode collection with buffers.
        experience_buffer_size: Size of experience buffer if using collection.
        episode_buffer_size: Size of episode buffer if using collection.
        save_episode_buffer: Optional path to save episode buffer as JSON.

    Returns:
        dict with "obs", "rewards", "info" from the final step, "history",
        and optionally "episode" (Episode object) if use_experience_collection is True.
    """
    # Use experience collection if requested
    if use_experience_collection:
        return _run_diplomacy_with_experience_collection(
            max_phases=max_phases,
            model=model,
            seed=seed,
            verbose=verbose,
            visualizer_url=visualizer_url,
            experience_buffer_size=experience_buffer_size,
            episode_buffer_size=episode_buffer_size,
            save_episode_buffer=save_episode_buffer,
        )
    
    # Original implementation
    env = DiplomacyNLWrapper(seed=seed)
    obs, info = env.reset()
    history = []
    rewards: Dict[str, float] = {}
    step_count = 0

    if visualizer_url:
        _push_visualizer_state(visualizer_url, _diplomacy_visualizer_state(env, info, status="running"))
        if verbose:
            print(f"[visualizer] Pushed initial state to {visualizer_url}")

    while not env.done and step_count < max_phases:
        # Dummy agent: for each power with an observation, call LLM to get orders
        actions: Dict[str, Union[List[str], str]] = {}
        for power_name, state_nl in obs.items():
            if not state_nl:
                continue
            action = language_agent_action(
                state_nl,
                game=GAME_DIPLOMACY,
                model=model,
                use_function_call=True,
                temperature=0.3,
            )
            if isinstance(action, list):
                actions[power_name] = action
            else:
                actions[power_name] = [action] if action else []
            if verbose:
                orders_preview = actions[power_name][:3] if isinstance(actions[power_name], list) else []
                print(f"  {power_name}: {len(actions[power_name]) if isinstance(actions[power_name], list) else 0} orders, e.g. {orders_preview}")

        obs, rewards, terminated, truncated, info = env.step(actions)
        history.append({"obs": obs, "rewards": dict(rewards), "info": dict(info)})
        step_count += 1

        if visualizer_url:
            status = "finished" if env.done else "running"
            _push_visualizer_state(visualizer_url, _diplomacy_visualizer_state(env, info, status=status))

        if verbose:
            phase = info.get("phase", "")
            print(f"Step {step_count} | Phase: {phase} | Done: {terminated}")

    if visualizer_url and env.done:
        _push_visualizer_state(visualizer_url, _diplomacy_visualizer_state(env, info, status="finished"))

    if verbose and env.done:
        print(f"Game over. Rewards (SC count / 18): {rewards}")

    return {
        "obs": obs,
        "rewards": rewards,
        "info": info,
        "history": history,
        "steps": step_count,
    }


def _run_diplomacy_with_experience_collection(
    max_phases: int = 5,
    model: str = "gpt-4o-mini",
    seed: int = 42,
    verbose: bool = True,
    visualizer_url: Optional[str] = None,
    experience_buffer_size: int = 10000,
    episode_buffer_size: int = 1000,
    save_episode_buffer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Diplomacy game with experience and episode collection.
    """
    env = DiplomacyNLWrapper(seed=seed)
    
    # Create buffer manager
    buffer_manager = AgentBufferManager(
        experience_buffer_size=experience_buffer_size,
        episode_buffer_size=episode_buffer_size,
    )
    
    # Run episode with experience collection
    task = "Diplomacy game"
    episode = buffer_manager.run_episode(
        env=env,
        task=task,
        game=GAME_DIPLOMACY,
        model=model,
        use_function_call=True,
        temperature=0.3,
        max_steps=max_phases,
        verbose=verbose,
    )
    
    # Save episode buffer if requested
    if save_episode_buffer:
        buffer_manager.save_episode_buffer(save_episode_buffer)
        if verbose:
            print(f"Saved episode buffer to {save_episode_buffer}")
    
    # Get final state info
    final_obs = episode.experiences[-1].next_state if episode.experiences else ""
    final_reward = episode.get_reward()
    final_info = {"episode_length": len(episode.experiences), "task": episode.task}
    
    # Build history from experiences
    history = []
    for exp in episode.experiences:
        history.append({
            "state": exp.state,
            "action": exp.action,
            "reward": exp.reward,
            "next_state": exp.next_state,
            "done": exp.done,
        })
    
    return {
        "obs": final_obs,
        "rewards": final_reward,
        "info": final_info,
        "history": history,
        "steps": len(episode.experiences),
        "episode": episode,
        "buffer_stats": buffer_manager.get_buffer_stats(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Diplomacy with dummy LLM agent for all powers")
    parser.add_argument("--max_phases", type=int, default=5, help="Max phases to run (each phase = one full turn)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument("--visualizer", action="store_true", help="Push state to games web server so you can watch rollout in browser")
    parser.add_argument("--visualizer_url", default="http://localhost:8000", help="Base URL of games web server (default: http://localhost:8000)")
    parser.add_argument("--use_experience_collection", action="store_true", help="Use experience/episode collection with buffers")
    parser.add_argument("--experience_buffer_size", type=int, default=10000, help="Size of experience buffer (default: 10000)")
    parser.add_argument("--episode_buffer_size", type=int, default=1000, help="Size of episode buffer (default: 1000)")
    parser.add_argument("--save_episode_buffer", type=str, default=None, help="Path to save episode buffer as JSON file")
    args = parser.parse_args()

    result = run_diplomacy_with_dummy_agent(
        max_phases=args.max_phases,
        model=args.model,
        seed=args.seed,
        verbose=not args.quiet,
        visualizer_url=args.visualizer_url if args.visualizer else None,
        use_experience_collection=args.use_experience_collection,
        experience_buffer_size=args.experience_buffer_size,
        episode_buffer_size=args.episode_buffer_size,
        save_episode_buffer=args.save_episode_buffer,
    )
    
    if args.use_experience_collection:
        print(f"Finished in {result['steps']} phases.")
        print(f"Episode task: {result['info'].get('task', 'N/A')}")
        print(f"Buffer stats: {result.get('buffer_stats', {})}")
    else:
        print(f"Finished in {result['steps']} phases. Game done: {result['info'].get('is_game_done')}")


if __name__ == "__main__":
    main()
