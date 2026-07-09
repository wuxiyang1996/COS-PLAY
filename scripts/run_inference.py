#!/usr/bin/env python
"""
Inference runner: play game environments using a trained Decision Agent
with a stored Skill Bank.

Loads a Decision Agent (local checkpoint or API model) and a persisted
Skill Bank, then runs episodes across supported game environments and
writes rollouts + aggregate metrics.

Usage (from Game-AI-Agent root):

    export PYTHONPATH="$(pwd):$PYTHONPATH"

    # Run with a co-evolution checkpoint and the latest skill bank
    python -m scripts.run_inference \
        --model runs/coevolution/models/decision_v3/global_step_20/actor/huggingface \
        --bank  runs/coevolution/skillbank/bank.jsonl \
        --games twenty_forty_eight sokoban \
        --episodes 10 --max-steps 200 \
        --output-dir runs/inference_results

    # Run with an API model and a bank snapshot directory
    python -m scripts.run_inference \
        --model gpt-4o-mini \
        --bank  runs/coevolution/skillbank \
        --episodes 5 --verbose

    # VERL-based inference (delegates to inference.run_verl_inference)
    python -m scripts.run_inference --verl [extra overrides...]

    # Evaluate a specific co-evolution iteration
    python -m scripts.run_inference \
        --coevo-dir runs/coevolution \
        --iteration 3 \
        --episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GAMINGAGENT_ROOT = REPO_ROOT.parent / "GamingAgent"

for p in [str(REPO_ROOT), str(GAMINGAGENT_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer

from decision_agents.agent import (
    VLMDecisionAgent,
    run_episode_vlm_agent,
)
from decision_agents.reward_func import RewardConfig
from decision_agents.agent_helper import EpisodicMemoryStore

from skill_agents.skill_bank.bank import SkillBankMVP

try:
    from skill_agents.query import SkillQueryEngine
except ImportError:
    SkillQueryEngine = None

from inference.run_decision_agent import run_inference as _run_single_episode

# ---------------------------------------------------------------------------
# Game environment registry (mirrors cold_start/generate_cold_start.py)
# ---------------------------------------------------------------------------
try:
    from cold_start.generate_cold_start import GAME_REGISTRY, ColdStartEnvWrapper
except ImportError:
    GAME_REGISTRY = {}
    ColdStartEnvWrapper = None

# Fallback: construct the registry from individual env imports if cold_start
# is not available.
if not GAME_REGISTRY:
    _ENVS: Dict[str, Any] = {}
    try:
        from gamingagent.envs.custom_01_2048.twentyFortyEightEnv import TwentyFortyEightEnv
        _ENVS["twenty_forty_eight"] = TwentyFortyEightEnv
    except ImportError:
        pass
    try:
        from gamingagent.envs.custom_02_sokoban.sokobanEnv import SokobanEnv
        _ENVS["sokoban"] = SokobanEnv
    except ImportError:
        pass
    try:
        from gamingagent.envs.custom_03_candy_crush.candyCrushEnv import CandyCrushEnv
        _ENVS["candy_crush"] = CandyCrushEnv
    except ImportError:
        pass
    try:
        from gamingagent.envs.custom_04_tetris.tetrisEnv import TetrisEnv
        _ENVS["tetris"] = TetrisEnv
    except ImportError:
        pass
    try:
        from gamingagent.envs.custom_05_doom.doomEnv import DoomEnvWrapper as DoomEnv
        _ENVS["doom"] = DoomEnv
    except ImportError:
        pass
    try:
        from gamingagent.envs.custom_06_pokemon_red.pokemonRedEnv import PokemonRedEnv
        _ENVS["pokemon_red"] = PokemonRedEnv
    except ImportError:
        pass
    GAME_REGISTRY = _ENVS


# ---------------------------------------------------------------------------
# Skill Bank loading
# ---------------------------------------------------------------------------

def load_skill_bank(
    bank_path: str,
    *,
    use_query_engine: bool = True,
) -> Tuple[SkillBankMVP, Any]:
    """Load a SkillBankMVP from a JSONL file or directory.

    If *bank_path* is a directory, looks for ``bank.jsonl``,
    ``skill_bank.jsonl``, or the most recent ``.jsonl`` file inside it.

    Returns (bank, query_engine_or_None).
    """
    bp = Path(bank_path)
    if bp.is_dir():
        candidates = ["bank.jsonl", "skill_bank.jsonl"]
        jsonl = None
        for c in candidates:
            if (bp / c).exists():
                jsonl = bp / c
                break
        if jsonl is None:
            jsonls = sorted(bp.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
            jsonl = jsonls[0] if jsonls else None
        if jsonl is None:
            print(f"[load_skill_bank] WARNING: no .jsonl found in {bp}, using empty bank.")
            return SkillBankMVP(), None
        bp = jsonl

    bank = SkillBankMVP(path=str(bp))
    bank.load()
    print(f"[load_skill_bank] Loaded {len(bank)} skills from {bp}")

    engine = None
    if use_query_engine and SkillQueryEngine is not None and len(bank) > 0:
        try:
            engine = SkillQueryEngine(bank)
            print(f"[load_skill_bank] SkillQueryEngine initialised (embedder={engine.has_embedder})")
        except Exception as exc:
            print(f"[load_skill_bank] SkillQueryEngine init failed: {exc}")
    return bank, engine


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(game_name: str, **kwargs: Any) -> Any:
    """Create a game environment by name.

    Tries GAME_REGISTRY first (from cold_start), then falls back to the
    env_wrappers module.
    """
    if ColdStartEnvWrapper is not None and game_name in GAME_REGISTRY:
        env_cls = GAME_REGISTRY[game_name]
        raw_env = env_cls(**kwargs) if kwargs else env_cls()
        return ColdStartEnvWrapper(raw_env, game=game_name)

    if game_name in GAME_REGISTRY:
        env_cls = GAME_REGISTRY[game_name]
        return env_cls(**kwargs) if kwargs else env_cls()

    # Try env_wrappers directly
    wrapper_map = {
        "overcooked": "OvercookedNLWrapper",
        "avalon": "AvalonNLWrapper",
        "diplomacy": "DiplomacyNLWrapper",
        "gamingagent": "GamingAgentNLWrapper",
        "videogamebench": "VideoGameBenchNLWrapper",
        "videogamebench_dos": "VideoGameBenchDOSNLWrapper",
    }
    if game_name in wrapper_map:
        import env_wrappers
        cls = getattr(env_wrappers, wrapper_map[game_name], None)
        if cls is not None:
            return cls(**kwargs) if kwargs else cls()

    raise ValueError(
        f"Unknown game: {game_name}. "
        f"Available: {sorted(set(list(GAME_REGISTRY.keys()) + list(wrapper_map.keys())))}"
    )


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: Any,
    agent: VLMDecisionAgent,
    *,
    task: str = "",
    max_steps: int = 500,
    verbose: bool = False,
) -> Episode:
    """Run one inference episode and return an Episode."""
    return run_episode_vlm_agent(
        env,
        agent=agent,
        task=task,
        max_steps=max_steps,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Multi-episode batch runner
# ---------------------------------------------------------------------------

def run_batch(
    game_name: str,
    agent: VLMDecisionAgent,
    *,
    num_episodes: int = 10,
    max_steps: int = 500,
    task: str = "",
    verbose: bool = False,
    output_dir: Optional[Path] = None,
    env_kwargs: Optional[dict] = None,
) -> List[Episode]:
    """Run *num_episodes* on *game_name* and return collected episodes.

    Saves per-episode JSON and an append-friendly JSONL to *output_dir*.
    """
    episodes: List[Episode] = []
    env_kw = env_kwargs or {}

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "rollouts.jsonl"
    else:
        jsonl_path = None

    for ep_idx in range(num_episodes):
        # Skip if episode already exists (resume support)
        if output_dir is not None:
            ep_file = output_dir / f"episode_{ep_idx:04d}.json"
            if ep_file.exists():
                print(f"[{game_name}] Episode {ep_idx} already exists, loading...")
                with open(ep_file) as f:
                    episode = Episode.from_dict(json.load(f))
                episodes.append(episode)
                continue

        t0 = time.time()
        try:
            env = make_env(game_name, **env_kw)
            episode = run_episode(
                env,
                agent,
                task=task or f"Play {game_name}",
                max_steps=max_steps,
                verbose=verbose,
            )
            elapsed = time.time() - t0

            n_steps = episode.get_length()
            total_r = episode.get_reward()
            total_shaped = episode.get_total_reward()
            done = episode.outcome if episode.outcome is not None else False

            print(
                f"[{game_name}] Episode {ep_idx}/{num_episodes}: "
                f"steps={n_steps}, r_env={total_r:.3f}, "
                f"r_total={total_shaped:.3f}, done={done}, "
                f"time={elapsed:.1f}s"
            )

            episodes.append(episode)

            # Persist
            if output_dir is not None:
                ep_dict = episode.to_dict()
                with open(ep_file, "w", encoding="utf-8") as f:
                    json.dump(ep_dict, f, default=str, ensure_ascii=False, indent=2)
                if jsonl_path is not None:
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(ep_dict, default=str, ensure_ascii=False) + "\n")

        except Exception:
            print(f"[{game_name}] Episode {ep_idx} FAILED:")
            traceback.print_exc()
            continue
        finally:
            try:
                if hasattr(env, "close"):
                    env.close()
            except Exception:
                pass

    return episodes


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def compute_metrics(episodes: List[Episode], game_name: str = "") -> Dict[str, Any]:
    """Aggregate per-game metrics from a list of episodes."""
    if not episodes:
        return {"game": game_name, "n_episodes": 0}

    rewards = [ep.get_reward() for ep in episodes]
    shaped_rewards = [ep.get_total_reward() for ep in episodes]
    lengths = [ep.get_length() for ep in episodes]
    completion_rate = sum(1 for ep in episodes if ep.outcome) / len(episodes)

    # Skill usage stats from experience action_types
    skill_queries = 0
    skill_calls = 0
    mem_queries = 0
    total_actions = 0
    for ep in episodes:
        for exp in ep.experiences:
            total_actions += 1
            at = getattr(exp, "action_type", None) or "primitive"
            if "QUERY_SKILL" in at.upper():
                skill_queries += 1
            elif "CALL_SKILL" in at.upper():
                skill_calls += 1
            elif "QUERY_MEM" in at.upper():
                mem_queries += 1

    return {
        "game": game_name,
        "n_episodes": len(episodes),
        "mean_reward": statistics.mean(rewards),
        "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_shaped_reward": statistics.mean(shaped_rewards),
        "mean_length": statistics.mean(lengths),
        "completion_rate": completion_rate,
        "total_actions": total_actions,
        "skill_queries": skill_queries,
        "skill_calls": skill_calls,
        "mem_queries": mem_queries,
        "skill_usage_rate": (skill_queries + skill_calls) / max(total_actions, 1),
    }


# ---------------------------------------------------------------------------
# Co-evolution checkpoint resolver
# ---------------------------------------------------------------------------

def resolve_coevo_checkpoint(
    coevo_dir: str,
    iteration: int,
    train_steps: int = 20,
    model_abbr: str = "CoEvo-Decision14B-SkillBank8B",
) -> Tuple[str, str]:
    """Resolve model and bank paths from a co-evolution output directory.

    Returns (model_path, bank_path).
    """
    base = Path(coevo_dir)
    model_path = (
        base / "models" / f"{model_abbr}_decision_v{iteration}"
        / f"global_step_{train_steps}" / "actor" / "huggingface"
    )
    bank_path = base / "skillbank"

    if not model_path.exists():
        # Try without model_abbr prefix
        for d in sorted((base / "models").glob(f"*decision_v{iteration}*")):
            candidate = d / f"global_step_{train_steps}" / "actor" / "huggingface"
            if candidate.exists():
                model_path = candidate
                break

    return str(model_path), str(bank_path)


# ---------------------------------------------------------------------------
# VERL inference delegate
# ---------------------------------------------------------------------------

def run_verl_inference(extra_overrides: list[str]) -> int:
    """Delegate to inference.run_verl_inference."""
    from inference.run_verl_inference import run_verl_inference as _verl
    return _verl(extra_overrides)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference with a trained Decision Agent and Skill Bank on game environments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--verl", action="store_true",
                       help="Use VERL-based inference (vLLM/sglang). Extra args are forwarded as Hydra overrides.")
    mode.add_argument("--local", action="store_true", default=True,
                       help="Use local inference with VLMDecisionAgent (default).")

    # Model & bank
    p.add_argument("--model", type=str, default=None,
                    help="Decision Agent model path (HF checkpoint) or API model name (e.g. gpt-4o-mini).")
    p.add_argument("--bank", type=str, default=None,
                    help="Path to skill bank JSONL file or directory containing bank snapshots.")
    p.add_argument("--no-bank", action="store_true",
                    help="Run without a skill bank (baseline / ablation).")

    # Co-evolution shortcut
    p.add_argument("--coevo-dir", type=str, default=None,
                    help="Co-evolution output directory (auto-resolves model and bank paths).")
    p.add_argument("--iteration", type=int, default=None,
                    help="Co-evolution iteration to evaluate (used with --coevo-dir).")
    p.add_argument("--train-steps", type=int, default=20,
                    help="Training steps per iteration (for checkpoint path resolution).")
    p.add_argument("--model-abbr", type=str, default="CoEvo-Decision14B-SkillBank8B",
                    help="Experiment abbreviation (for checkpoint path resolution).")

    # Games
    p.add_argument("--games", nargs="+", default=None,
                    help="Game names to evaluate (e.g. twenty_forty_eight sokoban). "
                         "Omit to run all available games.")
    p.add_argument("--list-games", action="store_true",
                    help="List available games and exit.")

    # Episode params
    p.add_argument("--episodes", type=int, default=10,
                    help="Number of episodes per game.")
    p.add_argument("--max-steps", type=int, default=500,
                    help="Maximum steps per episode.")
    p.add_argument("--task", type=str, default="",
                    help="Task description to pass to the agent.")

    # Output
    p.add_argument("--output-dir", type=str, default=None,
                    help="Directory to save rollouts and metrics. "
                         "Default: runs/inference_results/<timestamp>")
    p.add_argument("--verbose", "-v", action="store_true",
                    help="Print per-step agent actions and rewards.")

    # Agent hyperparameters
    p.add_argument("--retrieval-budget", type=int, default=10,
                    help="Steps between retrieval queries (retrieval_budget_n).")
    p.add_argument("--skill-abort-k", type=int, default=5,
                    help="Steps without progress before aborting active skill.")
    p.add_argument("--no-query-engine", action="store_true",
                    help="Disable SkillQueryEngine (use plain SkillBankMVP for skill queries).")

    # Reward config
    p.add_argument("--w-follow", type=float, default=0.1,
                    help="Weight on skill-following shaping reward.")
    p.add_argument("--query-cost", type=float, default=-0.05,
                    help="Cost per skill/memory query.")

    return p.parse_known_args()


def main() -> int:
    args, extra = parse_args()

    # -- List games and exit --
    if args.list_games:
        available = sorted(GAME_REGISTRY.keys())
        print("Available games:")
        for g in available:
            print(f"  - {g}")
        if not available:
            print("  (none — GamingAgent envs not found on PYTHONPATH)")
        print("\nEnv-wrapper games (require underlying env):")
        for g in ["overcooked", "avalon", "diplomacy", "gamingagent",
                   "videogamebench", "videogamebench_dos"]:
            print(f"  - {g}")
        return 0

    # -- VERL mode --
    if args.verl:
        return run_verl_inference(extra)

    # -- Resolve model & bank paths --
    model_path = args.model
    bank_path = args.bank

    if args.coevo_dir is not None:
        iteration = args.iteration
        if iteration is None:
            # Find the latest iteration
            models_dir = Path(args.coevo_dir) / "models"
            if models_dir.exists():
                versions = []
                for d in models_dir.iterdir():
                    if d.is_dir() and "_decision_v" in d.name:
                        try:
                            v = int(d.name.split("_decision_v")[-1])
                            versions.append(v)
                        except ValueError:
                            pass
                iteration = max(versions) if versions else 1
            else:
                iteration = 1
            print(f"[inference] Auto-detected latest iteration: v{iteration}")

        resolved_model, resolved_bank = resolve_coevo_checkpoint(
            args.coevo_dir, iteration,
            train_steps=args.train_steps,
            model_abbr=args.model_abbr,
        )
        if model_path is None:
            model_path = resolved_model
        if bank_path is None:
            bank_path = resolved_bank

    if model_path is None:
        model_path = "gpt-4o-mini"
        print(f"[inference] No --model specified, defaulting to {model_path}")

    # -- Load skill bank --
    bank = None
    query_engine = None
    if not args.no_bank and bank_path is not None:
        bank, query_engine = load_skill_bank(
            bank_path,
            use_query_engine=not args.no_query_engine,
        )
    elif args.no_bank:
        print("[inference] Running without skill bank (--no-bank).")

    # Use the query engine as the skill_bank if available (richer retrieval).
    skill_bank_for_agent = query_engine if query_engine is not None else bank

    # -- Build reward config --
    reward_config = RewardConfig(
        w_follow=args.w_follow,
        query_mem_cost=args.query_cost,
        query_skill_cost=args.query_cost,
    )

    # -- Build agent --
    agent = VLMDecisionAgent(
        model=model_path,
        skill_bank=skill_bank_for_agent,
        retrieval_budget_n=args.retrieval_budget,
        skill_abort_k=args.skill_abort_k,
        reward_config=reward_config,
    )
    print(f"[inference] Decision Agent: model={model_path}")
    print(f"[inference] Skill Bank: {len(bank) if bank else 0} skills")

    # -- Output directory --
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "runs" / "inference_results" / ts
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[inference] Output: {output_root}")

    # -- Determine games --
    if args.games:
        games = args.games
    else:
        games = sorted(GAME_REGISTRY.keys()) if GAME_REGISTRY else []
        if not games:
            print("[inference] ERROR: No games available. Specify --games or install GamingAgent envs.")
            return 1

    print(f"[inference] Games: {games}")
    print(f"[inference] Episodes per game: {args.episodes}, max steps: {args.max_steps}")
    print()

    # -- Run inference --
    all_metrics: List[Dict[str, Any]] = []
    all_episodes: Dict[str, List[Episode]] = {}
    t_start = time.time()

    for game_name in games:
        print(f"{'='*60}")
        print(f"  Game: {game_name}")
        print(f"{'='*60}")
        game_output = output_root / game_name

        episodes = run_batch(
            game_name,
            agent,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            task=args.task,
            verbose=args.verbose,
            output_dir=game_output,
        )

        metrics = compute_metrics(episodes, game_name)
        all_metrics.append(metrics)
        all_episodes[game_name] = episodes

        print(f"\n[{game_name}] Results:")
        print(f"  Episodes: {metrics['n_episodes']}")
        if metrics["n_episodes"] > 0:
            print(f"  Mean reward (env):    {metrics['mean_reward']:.3f} +/- {metrics['std_reward']:.3f}")
            print(f"  Mean reward (shaped): {metrics['mean_shaped_reward']:.3f}")
            print(f"  Mean length:          {metrics['mean_length']:.1f}")
            print(f"  Completion rate:      {metrics['completion_rate']:.1%}")
            print(f"  Skill usage rate:     {metrics['skill_usage_rate']:.1%}")
            print(f"    queries={metrics['skill_queries']}, "
                  f"calls={metrics['skill_calls']}, "
                  f"mem_queries={metrics['mem_queries']}")

        # Save per-game metrics
        with open(game_output / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        print()

    total_time = time.time() - t_start

    # -- Save aggregate results --
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "bank_path": bank_path,
        "bank_size": len(bank) if bank else 0,
        "games": games,
        "episodes_per_game": args.episodes,
        "max_steps": args.max_steps,
        "total_time_s": round(total_time, 1),
        "per_game": all_metrics,
    }

    # Cross-game aggregate
    all_rewards = [m["mean_reward"] for m in all_metrics if m["n_episodes"] > 0]
    if all_rewards:
        summary["overall_mean_reward"] = statistics.mean(all_rewards)
        summary["overall_completion_rate"] = statistics.mean(
            m["completion_rate"] for m in all_metrics if m["n_episodes"] > 0
        )
        summary["overall_skill_usage_rate"] = statistics.mean(
            m["skill_usage_rate"] for m in all_metrics if m["n_episodes"] > 0
        )

    summary_path = output_root / "inference_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # -- Print final summary --
    print(f"{'='*60}")
    print(f"  Inference Complete")
    print(f"{'='*60}")
    print(f"  Model:            {model_path}")
    print(f"  Skill Bank:       {len(bank) if bank else 0} skills")
    print(f"  Games evaluated:  {len(games)}")
    total_eps = sum(m["n_episodes"] for m in all_metrics)
    print(f"  Total episodes:   {total_eps}")
    print(f"  Wall time:        {total_time:.1f}s")
    if all_rewards:
        print(f"  Overall mean r:   {summary['overall_mean_reward']:.3f}")
        print(f"  Overall complete: {summary['overall_completion_rate']:.1%}")
        print(f"  Overall skill %:  {summary['overall_skill_usage_rate']:.1%}")
    print(f"  Results saved to: {output_root}")
    print(f"  Summary:          {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
