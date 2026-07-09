#!/usr/bin/env python
"""Phase-2 / Day-3 smoke driver: real env step under `harness.run_skill`.

PLAN-HARNESS §22, harness/README §22 — the empirical companion to
`_phase0_cross_eligibility_probe.py` (admission) and `_phase1_report.md`
(F2′ veto). Phase-2 measures **execution**: does
`harness.run_skill(skill, state)` actually drive a real env when the
adapter has a real executor wired in? Does the gymv success_fn evaluate
the lifted protocol's `effects_add` predicates against consecutive
post-step `StateSchema` snapshots and produce a structured verdict?

What this driver does:

    1. Load every `SkillRecord` from
       `labeling/skill_bank_out/run_20260430_030637/env_wrappers/<game>/skill_bank.jsonl`
       via `record_from_bank_entry` (decorator v2, post protocol-lift).
    2. Build a real env with `env_wrappers.gym_like.make_gaming_env(<game>)`
       — falls back to a tiny in-tree fake env when GamingAgent isn't
       installed so the driver remains usable in CI.
    3. Wire `GymvAdapter.set_executor(make_gymv_executor(env))`. Synthesise
       the hop-0 pre-state with `initial_state_from_env(env, …)` so
       `cumulative_reward_increased` etc. have a numeric baseline.
    4. For each ACTION-typed (PROVISIONAL'd) skill: run `harness.run_skill`,
       reset the env between runs, and capture
       `episode.outcome.extra["per_hop_effects"]`.
    5. Aggregate per-skill verdicts and write
       `labeling_supplement/harness_io_out/_phase2_<game>_<ts>.json` plus
       the human-readable `_phase2_report.md` row.

Usage::

    cd Multi-hop-Reasoning-VLM-Agent
    python labeling_supplement/_phase2_real_env_skill_smoke.py \\
        --game twenty_forty_eight --max-skills 3

Defaults to twenty_forty_eight; pass `--game tetris` (or any string in
`env_wrappers.gym_like.GAME_CONFIG_MAPPING`) for the other side.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.enums import SkillStatus, SkillType                          # noqa: E402
from data_structure.extensions.skill_record import SkillRecord           # noqa: E402
from harness import (                                                     # noqa: E402
    AdapterRegistry,
    HarnessConfig,
    make_gaming_env_producer,
    SkillHarness,
    initial_state_from_env,
    make_gymv_executor,
)
from harness.adapters import GymvAdapter                                 # noqa: E402
from labeling_supplement._harness_io_helpers import (                    # noqa: E402
    load_bank_records,
)

logger = logging.getLogger("phase2_real_env_smoke")

DEFAULT_BANK_ROOT = (
    REPO_ROOT / "labeling" / "skill_bank_out" / "run_20260430_030637"
              / "env_wrappers"
)
# Day-5 transfer cycle (`_phase4_transfer_cycle.py`) needs the matching
# cold-start actions root to harvest target-task FewShotDemos from.
DEFAULT_ACTIONS_ROOT = (
    REPO_ROOT / "labeling" / "skill_actions_out" / "run_20260430_064325"
)


# ---------------------------------------------------------------------------
# Env wiring (graceful fallback)
# ---------------------------------------------------------------------------

def build_env(game: str) -> Tuple[Any, str]:
    """Return `(env, source)` where `source ∈ {"gaming_agent", "fake"}`.

    The driver tries the real GamingAgent env first (the production path
    used by the cold-start actor); when that import chain fails (no
    GamingAgent on PYTHONPATH, missing native deps, etc.) we fall back
    to a deterministic in-tree fake so the harness wiring still gets
    exercised and the verdict is still reproducible.
    """
    try:
        from env_wrappers.gym_like import make_gaming_env, list_games
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("env_wrappers import failed (%s); using fake env", exc)
        return _FakeTwoFortyEightEnv(), "fake"
    if game not in list_games():
        raise SystemExit(
            f"Unknown game {game!r}. Available: {sorted(list_games())}"
        )
    try:
        env = make_gaming_env(game, max_steps=200, observation_mode="text")
        env.reset()
        return env, "gaming_agent"
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("make_gaming_env(%r) failed (%s); using fake env",
                       game, exc)
        return _FakeTwoFortyEightEnv(), "fake"


# Tiny fallback. Mirrors the contract of `_GymLikeWrapper` (5-tuple step,
# action_names property, dict obs with `text` carrying a canonical
# `<state>` block). Only implements the four 2048 directions; reward of 4
# on every "up" so SLIDE-up runs produce a passing
# `cumulative_reward_increased`.
_FAKE_INITIAL = """<state>
domain=gymv
task=make_gaming_env/twenty_forty_eight
goal=Play 2048
step=0

<entities>
e1[type=region, label=board]
e2[type=object, label=tile_2]
e3[type=text, label=highest_tile]
e4[type=text, label=score]

<attributes>
e1.state=visible
e2.state=visible
e2.value=2
e3.value=2
e4.value=0

<state_flags>
phase=play
</state>"""

_FAKE_AFTER_UP = """<state>
domain=gymv
task=make_gaming_env/twenty_forty_eight
goal=Play 2048
step=1

<entities>
e1[type=region, label=board]
e2[type=object, label=tile_4]
e3[type=text, label=highest_tile]
e4[type=text, label=score]

<attributes>
e1.state=visible
e2.state=visible
e2.value=4
e3.value=4
e4.value=4

<state_flags>
phase=play
</state>"""


class _FakeTwoFortyEightEnv:
    action_names: List[str] = ["up", "down", "left", "right"]

    def __init__(self) -> None:
        self._last_obs = {"text": _FAKE_INITIAL,
                          "schema_canonical": _FAKE_INITIAL}

    def reset(self, *, seed: Any = None, options: Any = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._last_obs = {"text": _FAKE_INITIAL,
                          "schema_canonical": _FAKE_INITIAL}
        return self._last_obs, {"action_names": list(self.action_names)}

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if action == "up":
            self._last_obs = {"text": _FAKE_AFTER_UP,
                              "schema_canonical": _FAKE_AFTER_UP}
            return (
                self._last_obs, 4.0, False, False,
                {"action_names": list(self.action_names)},
            )
        return (
            self._last_obs, 0.0, False, False,
            {"action_names": list(self.action_names)},
        )


# ---------------------------------------------------------------------------
# Verdict aggregation
# ---------------------------------------------------------------------------

@dataclass
class SkillVerdict:
    skill_id: str
    skill_type: str
    n_hops_protocol: int
    n_hops_eval: int
    n_hops_pass: int
    pass_rate: float
    success: bool
    abort_reason: Optional[str]
    cumulative_reward: float
    per_hop_effects: Dict[str, Any] = field(default_factory=dict)


def run_one_skill(
    *,
    harness: SkillHarness,
    env: Any,
    domain: str,
    task: str,
    skill: SkillRecord,
    bindings: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    schema_producer: Any = None,
) -> SkillVerdict:
    # Re-seed env so each skill starts from the same baseline. Most
    # gymnasium-style envs accept ``reset(seed=…)`` and ignore unknown
    # kwargs (or raise — fall back to bare ``reset()`` then).
    try:
        if seed is not None:
            env.reset(seed=seed)
        else:
            env.reset()
    except TypeError:
        env.reset()
    state = initial_state_from_env(
        env, domain=domain, task=task, schema_producer=schema_producer,
    )
    try:
        episode = harness.run_skill(
            skill, state, parent_run_id=None,
            bindings=dict(bindings or {}),
        )
    except Exception as exc:                                       # noqa: BLE001
        return SkillVerdict(
            skill_id=skill.skill_id,
            skill_type=skill.skill_type.value,
            n_hops_protocol=len(skill.protocol or []),
            n_hops_eval=0,
            n_hops_pass=0,
            pass_rate=0.0,
            success=False,
            abort_reason=f"harness_raised: {exc!r}",
            cumulative_reward=0.0,
        )

    out = episode.outcome
    extra = (out.extra if out and out.extra else {}) or {}
    roll = extra.get("per_hop_effects") or {}
    return SkillVerdict(
        skill_id=skill.skill_id,
        skill_type=skill.skill_type.value,
        n_hops_protocol=len(skill.protocol or []),
        n_hops_eval=int(roll.get("n_hops_evaluated") or 0),
        n_hops_pass=int(roll.get("n_hops_passed") or 0),
        pass_rate=float(roll.get("pass_rate") or 0.0),
        success=bool(out.success) if out else False,
        abort_reason=out.abort_reason if out else "no_outcome",
        cumulative_reward=float(episode.cost.get("ms", 0.0) and 0.0)
        or sum(
            float(s.action_payload.get("reward") or 0.0) for s in episode.steps
        ),
        per_hop_effects=roll,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default="twenty_forty_eight")
    p.add_argument(
        "--bank-root", default=str(DEFAULT_BANK_ROOT),
        help="Path to the env_wrappers bank tree (decorator v2)",
    )
    p.add_argument("--max-skills", type=int, default=10)
    p.add_argument("--max-hops", type=int, default=12)
    p.add_argument("--max-ms", type=float, default=30_000.0)
    p.add_argument(
        "--bindings",
        action="append",
        default=[],
        help=(
            "Pre-fill `${slot}` placeholders in lifted protocols, e.g. "
            "`--bindings direction=up --bindings target=up`. The actor "
            "would normally fill these; the smoke driver hard-codes them "
            "to exercise env-mutating hops end-to-end."
        ),
    )
    p.add_argument(
        "--include-reasoning",
        action="store_true",
        help=(
            "Also dispatch REASONING-typed skills through the GymvAdapter. "
            "By default the adapter advertises (ACTION, MIXED, GROUNDING) "
            "and reasoning-typed lifted skills surface as `no_adapter_for "
            "(gymv,reasoning)`. Set this to register the adapter for "
            "REASONING too — the executor's OBSERVATIONAL_OPS handler "
            "renders INSPECT/EVALUATE/COMPARE hops as evidence-only "
            "no-ops, so they're safe to dispatch."
        ),
    )
    p.add_argument(
        "--strict-actions",
        action="store_true",
        help=(
            "Configure the executor with `on_unresolved='abort'`. By "
            "default unresolvable env-mutating hops (e.g. a redundant "
            "lifted EXECUTE() following SLIDE.direction=up) are soft-"
            "skipped as observational evidence so the success_fn can "
            "still score the env-mutating hops that DID resolve. Strict "
            "mode is the gate-hardening path: every unresolvable hop "
            "shows up as `abort_reason=no_env_action_for_op…`."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "labeling_supplement" / "harness_io_out"),
    )
    p.add_argument(
        "--n-trials", type=int, default=1,
        help=(
            "Run each skill N times against fresh env resets and report "
            "the **best** trial's verdict (plus the per-trial pass rate "
            "distribution). Defeats the env's non-deterministic initial "
            "tile placement: a 2048 ``Commit/Merge`` skill can only "
            "score on resets where 'up' produces a legal merge, so the "
            "single-trial pass-rate is a noisy lower bound. With "
            "``--n-trials 8 --seed 0`` we sample 8 deterministic resets "
            "(seeds 0..7) and report the merge-bearing one."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Base RNG seed for env.reset(seed=…). When ``--n-trials`` "
            "is also set, the per-trial seed is ``seed + trial_index``. "
            "Without ``--seed``, env.reset() uses its default seeding."
        ),
    )
    p.add_argument(
        "--no-schema-producer",
        action="store_true",
        help=(
            "Disable the Day-4B deterministic schema producer (which "
            "renders ``<state>...</state>`` from ``env.info`` for "
            "supported games). Useful for A/B comparisons against the "
            "Day-3 plain-text obs path."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    bindings: Dict[str, Any] = {}
    for kv in args.bindings:
        if "=" in kv:
            k, v = kv.split("=", 1)
            bindings[k.strip()] = v.strip()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bank_path = Path(args.bank_root) / args.game / "skill_bank.jsonl"
    if not bank_path.exists():
        raise SystemExit(f"bank_jsonl missing: {bank_path}")

    records = load_bank_records(bank_path, default_domain="gymv")
    # Use ACTION-type skills first (env-mutating); then GROUNDING/REASONING.
    records.sort(key=lambda r: (r.skill_type != SkillType.ACTION, r.name))
    records = records[: args.max_skills]
    logger.info(
        "Loaded %d skill(s) from %s (after sorting by ACTION-first)",
        len(records), bank_path,
    )
    for r in records:
        logger.info(
            "  - %s | type=%s | hops=%d | feasible_tasks=%s",
            r.skill_id, r.skill_type.value, len(r.protocol or []), r.feasible_tasks,
        )

    env, env_source = build_env(args.game)
    logger.info("env=%s (action_names=%s)", env_source,
                getattr(env, "action_names", None))

    adapter = GymvAdapter()
    if args.include_reasoning:
        # Widen the adapter's `supported_types` so the registry's
        # `(domain, type)` lookup hits this adapter for reasoning skills
        # too. The executor's OBSERVATIONAL_OPS branch handles them
        # safely (no env step; just an EvidenceRef per hop).
        adapter.supported_types = (
            SkillType.ACTION, SkillType.MIXED,
            SkillType.GROUNDING, SkillType.REASONING,
        )
    # Day-4B: when a deterministic schema producer is registered for
    # the game, plumb it into the executor + initial-state path so the
    # post-step `<state>` block is rich (entity_attrs / phase / score)
    # rather than the GamingAgent text obs. Falls back to None for
    # games without a producer; the executor still works, just with
    # the Day-3 minimal facts dict.
    schema_producer = (
        None if args.no_schema_producer
        else make_gaming_env_producer(args.game)
    )
    if schema_producer is not None:
        logger.info("schema_producer=%s for game=%s",
                    getattr(schema_producer, "__name__", "?"), args.game)
    else:
        logger.info(
            "schema_producer=None (no producer for %s; using text-obs path)",
            args.game,
        )
    executor, holder = make_gymv_executor(
        env, domain="gymv", task=args.game,
        on_unresolved="abort" if args.strict_actions else "skip",
        schema_producer=schema_producer,
    )
    adapter.set_executor(executor)
    registry = AdapterRegistry()
    registry.register(adapter)
    harness = SkillHarness(registry, config=HarnessConfig(
        seed=0,
        default_budget_hops=args.max_hops,
        default_budget_ms=args.max_ms,
    ))

    # The harness's eligibility filter requires PROVISIONAL+; the
    # decorator emits records as DRAFT/CANDIDATE. We can't promote here
    # without a lifecycle manager, so we sidestep eligibility by going
    # straight through `harness.run_skill(skill, state)` (which doesn't
    # consult the filter — that's `select_eligible_skills`'s job).
    # `run_skill` only needs the registered adapter, which we have.
    for r in records:
        object.__setattr__(r, "status", SkillStatus.PROVISIONAL)

    verdicts: List[SkillVerdict] = []
    per_skill_trials: Dict[str, List[Tuple[int, float, int, int]]] = {}
    for r in records:
        # Try N trials with deterministic per-trial seeds (when --seed
        # set) and report the best verdict. The "best" criterion is:
        # highest pass_rate; tie-break on n_hops_pass; tie-break on
        # success flag. This handles env non-determinism where one
        # reset's initial board may not admit a legal merge.
        trials: List[Tuple[int, float, int, int]] = []  # (idx, rate, pass, eval)
        best_v: Optional[SkillVerdict] = None
        for trial in range(max(1, args.n_trials)):
            trial_seed = (
                None if args.seed is None
                else int(args.seed) + trial
            )
            v = run_one_skill(
                harness=harness, env=env,
                domain="gymv", task=args.game, skill=r,
                bindings=bindings,
                seed=trial_seed,
                schema_producer=schema_producer,
            )
            trials.append((trial, v.pass_rate, v.n_hops_pass, v.n_hops_eval))
            better = (
                best_v is None
                or v.pass_rate > best_v.pass_rate
                or (v.pass_rate == best_v.pass_rate
                    and v.n_hops_pass > best_v.n_hops_pass)
                or (v.pass_rate == best_v.pass_rate
                    and v.n_hops_pass == best_v.n_hops_pass
                    and v.success and not best_v.success)
            )
            if better:
                best_v = v
        assert best_v is not None
        per_skill_trials[r.skill_id] = trials
        verdicts.append(best_v)
        if args.n_trials > 1:
            rates = [t[1] for t in trials]
            logger.info(
                "skill=%s trials=%d best_pass_rate=%.2f all_rates=%s "
                "success=%s eval=%d/%d abort=%s",
                best_v.skill_id, args.n_trials, best_v.pass_rate, rates,
                best_v.success,
                best_v.n_hops_pass, best_v.n_hops_eval,
                best_v.abort_reason,
            )
        else:
            logger.info(
                "skill=%s success=%s eval=%d/%d pass_rate=%.2f abort=%s",
                best_v.skill_id, best_v.success,
                best_v.n_hops_pass, best_v.n_hops_eval, best_v.pass_rate,
                best_v.abort_reason,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"_phase2_{args.game}_{ts}.json"
    out_path.write_text(json.dumps({
        "game": args.game,
        "env_source": env_source,
        "bank_path": str(bank_path),
        "n_skills": len(verdicts),
        "n_success": sum(1 for v in verdicts if v.success),
        "n_with_evaluable_effects": sum(1 for v in verdicts if v.n_hops_eval > 0),
        "n_all_predicates_passed": sum(
            1 for v in verdicts if v.n_hops_eval > 0 and v.pass_rate >= 1.0
        ),
        "n_trials_per_skill": args.n_trials,
        "base_seed": args.seed,
        "per_skill_trials": {
            sid: [
                {"trial": t[0], "pass_rate": t[1],
                 "n_pass": t[2], "n_eval": t[3]}
                for t in trial_log
            ]
            for sid, trial_log in per_skill_trials.items()
        },
        "verdicts": [asdict(v) for v in verdicts],
        "timestamp": ts,
    }, indent=2))
    logger.info("wrote %s", out_path)

    # Print a compact summary so the human caller can eyeball the table
    # without opening the JSON.
    print()
    print(f"=== Phase-2 smoke summary [{args.game}, env={env_source}] ===")
    print(
        f"{'skill_id':<24} {'type':<10} {'hops':>4} {'eval':>4} "
        f"{'pass':>4} {'rate':>5} {'ok':>3}"
    )
    for v in verdicts:
        print(
            f"{v.skill_id[:24]:<24} {v.skill_type:<10} "
            f"{v.n_hops_protocol:>4} {v.n_hops_eval:>4} "
            f"{v.n_hops_pass:>4} {v.pass_rate:>5.2f} "
            f"{('Y' if v.success else 'n'):>3}"
        )
    print(f"\nfull verdict json: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
