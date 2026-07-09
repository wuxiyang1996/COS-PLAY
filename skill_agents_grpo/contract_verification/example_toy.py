"""
Toy example demonstrating Stage 3 ContractVerification standalone.

Two-round scenario:
  Round 1: Build contracts from consistent data → KEEP.
  Round 2: New trajectory data with drift → REFINE (drop spurious literal).

Also demonstrates:
  - ``__NEW__`` materialization into a new skill.
  - ``run_stage3`` pipeline API.
  - Agent-facing summary (for LLM teacher / preference loop).
  - Bank providing ``compat_fn`` back to Stage 2.

Run:
    python -m skill_agents.contract_verification.example_toy
"""

from __future__ import annotations

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from skill_agents_grpo.contract_verification.config import ContractVerificationConfig
from skill_agents_grpo.contract_verification.schemas import SegmentRecord, SkillContract
from skill_agents_grpo.contract_verification.predicates import build_segment_predicates
from skill_agents_grpo.contract_verification.contract_init import (
    compute_all_effects,
    build_initial_contracts,
)
from skill_agents_grpo.contract_verification.contract_verify import (
    verify_all_contracts,
    verify_contract,
)
from skill_agents_grpo.contract_verification.updates import (
    decide_action,
    apply_refine,
    materialize_new_skills,
)
from skill_agents_grpo.contract_verification.skill_bank import SkillBank
from skill_agents_grpo.contract_verification.run_stage3 import Stage3Summary


def make_toy_observations(T: int) -> list:
    """Create T synthetic observation strings."""
    return [f"obs_{t}" for t in range(T)]


def toy_extract_predicates(obs: str) -> dict:
    """Deterministic predicate extractor keyed on timestep index.

    Simulates a world where:
      - ``has_key`` becomes True at t=10.
      - ``door_open`` becomes True at t=20.
      - ``near_chest`` is True for t in [30..49].
      - ``chest_open`` becomes True at t=35.
      - ``has_sword`` is True for t < 10.
      - ``fog`` is True sporadically (noise).
    """
    t = int(obs.split("_")[1])
    preds: dict = {}
    preds["has_key"] = 1.0 if t >= 10 else 0.0
    preds["door_open"] = 1.0 if t >= 20 else 0.0
    preds["near_chest"] = 1.0 if 30 <= t < 50 else 0.0
    preds["chest_open"] = 1.0 if t >= 35 else 0.0
    preds["has_sword"] = 1.0 if t < 10 else 0.0
    preds["fog"] = 1.0 if t % 7 == 0 else 0.0
    return preds


def _build_records(segments_info, observations, config, extract_fn):
    """Build SegmentRecord list from segment specs."""
    T = len(observations)
    records = []
    for seg_id, traj_id, t_start, t_end, skill in segments_info:
        rec = build_segment_predicates(
            seg_id=seg_id, traj_id=traj_id,
            t_start=t_start, t_end=min(t_end, T - 1),
            skill_label=skill, observations=observations,
            config=config.predicates, extract_fn=extract_fn,
        )
        records.append(rec)
    return records


def _print_section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def main() -> None:
    T = 50
    observations = make_toy_observations(T)
    config = ContractVerificationConfig()
    config.new_skill.min_effect_distinctiveness = 0.1
    bank = SkillBank()

    print("=" * 60)
    print("  Stage 3 Toy Example -- Two-Round Contract Verification")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════
    #  ROUND 1: Build initial contracts from consistent data
    # ══════════════════════════════════════════════════════════════
    _print_section("ROUND 1: Initial contract building")

    round1_segments = [
        ("r1_seg0", "traj1", 0,  9,  "get_key"),
        ("r1_seg1", "traj1", 5,  14, "get_key"),
        ("r1_seg2", "traj2", 0,  9,  "get_key"),
        ("r1_seg3", "traj1", 10, 19, "open_door"),
        ("r1_seg4", "traj1", 15, 24, "open_door"),
        ("r1_seg5", "traj2", 10, 19, "open_door"),
    ]

    records = _build_records(round1_segments, observations, config, toy_extract_predicates)
    compute_all_effects(records, config.predicates, config.aggregation)

    for rec in records:
        print(f"  {rec.seg_id}: [{rec.t_start}..{rec.t_end}] "
              f"skill={rec.skill_label}, "
              f"add={rec.effects_add or '{}'}, del={rec.effects_del or '{}'}")

    contracts = build_initial_contracts(records, config.predicates, config.aggregation)
    reports = verify_all_contracts(contracts, records, config)

    print("\nContracts learned:")
    for sid, c in contracts.items():
        r = reports[sid]
        print(f"  {sid} v{c.version}: pre={c.pre}, eff_add={c.eff_add}, "
              f"inv={c.inv} -- pass={r.overall_pass_rate:.2f}")
        bank.add_or_update(c, r)

    print(f"\nBank after Round 1: {bank.active_skill_ids}")

    # ══════════════════════════════════════════════════════════════
    #  ROUND 2: New data drifts — pre "has_sword" is no longer consistent
    #  for get_key, triggering REFINE.
    # ══════════════════════════════════════════════════════════════
    _print_section("ROUND 2: Drift detection and REFINE")

    # In this round some get_key instances start without has_sword
    # (simulated by using segments that start at t >= 10 where has_sword=0).
    # We also include __NEW__ segments that consistently gain chest_open.

    def drift_extract_predicates(obs: str) -> dict:
        """Modified extractor where has_sword is absent in later regions."""
        t = int(obs.split("_")[1])
        preds = toy_extract_predicates(obs)
        if t >= 3:
            preds["has_sword"] = 0.0
        return preds

    round2_segments = [
        ("r2_seg0", "traj3", 0,  9,  "get_key"),    # has_sword at start
        ("r2_seg1", "traj3", 3,  12, "get_key"),     # NO has_sword at start (drift)
        ("r2_seg2", "traj4", 4,  13, "get_key"),     # NO has_sword at start (drift)
        ("r2_seg3", "traj3", 10, 19, "open_door"),
        ("r2_seg4", "traj4", 10, 19, "open_door"),
        # NEW segments that consistently gain chest_open
        ("r2_new0", "traj3", 30, 39, "__NEW__"),
        ("r2_new1", "traj3", 31, 40, "__NEW__"),
        ("r2_new2", "traj4", 30, 39, "__NEW__"),
        ("r2_new3", "traj4", 31, 40, "__NEW__"),
        ("r2_new4", "traj4", 32, 41, "__NEW__"),
    ]

    records2 = _build_records(round2_segments, observations, config, drift_extract_predicates)
    compute_all_effects(records2, config.predicates, config.aggregation)

    existing_records2 = [r for r in records2 if r.skill_label != "__NEW__"]
    new_records2 = [r for r in records2 if r.skill_label == "__NEW__"]

    print("New segment effects:")
    for rec in records2:
        print(f"  {rec.seg_id}: [{rec.t_start}..{rec.t_end}] skill={rec.skill_label}, "
              f"add={rec.effects_add or '{}'}, del={rec.effects_del or '{}'}")

    # Verify the EXISTING contracts (from Round 1) against new data
    reports2 = verify_all_contracts(bank.active_contracts, existing_records2, config)

    print("\nVerification of Round 1 contracts against Round 2 data:")
    from collections import defaultdict
    by_skill = defaultdict(list)
    for rec in existing_records2:
        by_skill[rec.skill_label].append(rec)

    for sid in bank.active_skill_ids:
        r = reports2.get(sid)
        if r is None:
            continue
        contract = bank.get_contract(sid)
        print(f"  {sid}: pass={r.overall_pass_rate:.2f}, "
              f"pre_violations={r.pre_violation_rate}")

        instances = by_skill.get(sid, [])
        action = decide_action(contract, r, instances, config)
        print(f"    -> Action: {action.action}")

        if action.action == "REFINE":
            print(f"      dropped: {action.dropped_literals}")
            print(f"      demoted: {action.demoted_to_soft}")
            apply_refine(contract, action)
            print(f"      refined: pre={contract.pre}, v{contract.version}")
            bank.add_or_update(contract, r)
        elif action.action == "KEEP":
            bank.add_or_update(contract, r)

    # Materialize NEW
    if new_records2:
        new_contracts, new_actions, new_reports, resegment = materialize_new_skills(
            new_records2, bank.active_contracts, config,
        )
        print(f"\nNEW materialization: {len(new_contracts)} new skill(s), "
              f"resegment_needed={resegment}")
        for nc, nr in zip(new_contracts, new_reports):
            print(f"  {nc.skill_id}: eff_add={nc.eff_add}, eff_del={nc.eff_del}, "
                  f"pre={nc.pre}, pass={nr.overall_pass_rate:.2f}")
            bank.add_or_update(nc, nr)

    # ══════════════════════════════════════════════════════════════
    #  Final bank + Stage 2 integration demo
    # ══════════════════════════════════════════════════════════════
    _print_section("Final Skill Bank")

    for sid, info in bank.summary().items():
        pr = info.get("pass_rate", 0)
        print(f"  {sid} v{info['version']}: "
              f"pre={info['pre']}, eff_add={info['eff_add']}, "
              f"eff_del={info['eff_del']} -- pass={pr:.2f}")

    _print_section("Stage 2 Integration: compat_fn demo")

    test_cases = [
        ("get_key", {"has_sword": 0.0}, {"has_key": 1.0}),
        ("open_door", {"has_key": 1.0}, {"door_open": 1.0, "has_key": 1.0}),
        ("open_door", {"has_key": 0.0}, {"door_open": 1.0}),  # missing pre
    ]
    for skill, p_start, p_end in test_cases:
        score = bank.compat_fn(skill, p_start, p_end)
        print(f"  compat({skill}, start={p_start}, end={p_end})")
        print(f"    -> score = {score:.3f}")

    print(f"\n  Active skills for Stage 2: {bank.get_skill_names()}")

    # ══════════════════════════════════════════════════════════════
    #  Action language representations
    # ══════════════════════════════════════════════════════════════
    _print_section("Action Language: PDDL domain")
    print(bank.to_action_language(fmt="pddl", domain_name="toy_game"))

    _print_section("Action Language: STRIPS operators")
    print(bank.to_action_language(fmt="strips"))

    _print_section("Action Language: SAS+ operators")
    print(bank.to_action_language(fmt="sas"))

    _print_section("Action Language: Compact (for LLM context)")
    print(bank.to_action_language(fmt="compact"))

    _print_section("Per-contract method: open_door.to_action_language('pddl')")
    od_contract = bank.get_contract("open_door")
    if od_contract:
        print(od_contract.to_action_language("pddl"))

    _print_section("Agent-facing summary (LLM context)")
    print(f"  Bank has {len(bank)} skills, {len(bank.active_skill_ids)} active")
    print(f"  History: {len(bank.history)} events")

    print("\nDone.")


if __name__ == "__main__":
    main()
