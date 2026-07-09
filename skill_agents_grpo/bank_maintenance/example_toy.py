"""
Toy example exercising the full bank maintenance pipeline.

Creates a synthetic skill bank with:
  - A multi-modal skill that should be SPLIT.
  - A duplicate pair that should be MERGED.
  - A skill with fragile literals that should be REFINED (weakened).
  - A skill confused with another that should be REFINED (strengthened).

Then runs ``run_bank_maintenance`` and prints the diff report.
"""

from __future__ import annotations

import random
from collections import Counter
from pprint import pprint

from skill_agents_grpo.skill_bank.bank import SkillBankMVP
from skill_agents_grpo.stage3_mvp.schemas import (
    SegmentRecord,
    SkillEffectsContract,
    VerificationReport,
)
from skill_agents_grpo.bank_maintenance.config import BankMaintenanceConfig
from skill_agents_grpo.bank_maintenance.run_bank_maintenance import run_bank_maintenance

random.seed(42)


def _make_segment(
    seg_id: str,
    traj_id: str,
    skill: str,
    t_start: int,
    t_end: int,
    eff_add: set,
    eff_del: set,
    eff_event: set,
) -> SegmentRecord:
    return SegmentRecord(
        seg_id=seg_id,
        traj_id=traj_id,
        t_start=t_start,
        t_end=t_end,
        skill_label=skill,
        eff_add=set(eff_add),
        eff_del=set(eff_del),
        eff_event=set(eff_event),
        B_start=set(),
        B_end=set(eff_add),
    )


def build_toy_data():
    bank = SkillBankMVP()
    segments: list[SegmentRecord] = []
    seg_counter = 0

    def _sid():
        nonlocal seg_counter
        seg_counter += 1
        return f"seg_{seg_counter:04d}"

    # ── Skill A: multi-modal (should SPLIT) ──────────────────────
    mode1_add = {"p1", "p2", "p3"}
    mode2_add = {"p4", "p5", "p6"}

    for i in range(15):
        eff = mode1_add if i < 8 else mode2_add
        segments.append(_make_segment(
            _sid(), f"traj_{i % 3}", "skill_A",
            t_start=i * 100, t_end=i * 100 + 50,
            eff_add=eff, eff_del=set(), eff_event={"click"},
        ))

    contract_a = SkillEffectsContract(
        skill_id="skill_A",
        eff_add=mode1_add | mode2_add,
        eff_event={"click"},
        support={p: 8 for p in mode1_add} | {p: 7 for p in mode2_add} | {"click": 15},
        n_instances=15,
    )
    report_a = VerificationReport(
        skill_id="skill_A",
        n_instances=15,
        eff_add_success_rate={p: 0.53 for p in mode1_add | mode2_add},
        eff_event_rate={"click": 1.0},
        overall_pass_rate=0.40,
        failure_signatures={
            "miss_add:p4|miss_add:p5|miss_add:p6": 8,
            "miss_add:p1|miss_add:p2|miss_add:p3": 7,
        },
    )
    bank.add_or_update(contract_a, report_a)

    # ── Skill B & C: near-duplicates (should MERGE) ─────────────
    shared_effects = {"q1", "q2", "q3", "q4", "q5"}

    for label in ["skill_B", "skill_C"]:
        for i in range(12):
            noise = {f"q_noise_{random.randint(100,999)}"}
            segments.append(_make_segment(
                _sid(), f"traj_{i % 4}", label,
                t_start=2000 + i * 80, t_end=2000 + i * 80 + 40,
                eff_add=shared_effects | (noise if random.random() < 0.1 else set()),
                eff_del=set(), eff_event={"submit"},
            ))

    for label in ["skill_B", "skill_C"]:
        c = SkillEffectsContract(
            skill_id=label,
            eff_add=shared_effects,
            eff_event={"submit"},
            support={p: 12 for p in shared_effects} | {"submit": 12},
            n_instances=12,
        )
        r = VerificationReport(
            skill_id=label,
            n_instances=12,
            eff_add_success_rate={p: 0.95 for p in shared_effects},
            eff_event_rate={"submit": 1.0},
            overall_pass_rate=0.92,
        )
        bank.add_or_update(c, r)

    # ── Skill D: fragile contract (should REFINE / weaken) ───────
    good_preds = {"r1", "r2", "r3"}
    fragile_preds = {"r_fragile_1", "r_fragile_2"}

    for i in range(20):
        eff = set(good_preds)
        if random.random() > 0.3:
            eff |= fragile_preds
        segments.append(_make_segment(
            _sid(), f"traj_{i % 5}", "skill_D",
            t_start=4000 + i * 60, t_end=4000 + i * 60 + 30,
            eff_add=eff, eff_del={"r_del"}, eff_event=set(),
        ))

    contract_d = SkillEffectsContract(
        skill_id="skill_D",
        eff_add=good_preds | fragile_preds,
        eff_del={"r_del"},
        support={p: 20 for p in good_preds} | {p: 14 for p in fragile_preds} | {"r_del": 20},
        n_instances=20,
    )
    report_d = VerificationReport(
        skill_id="skill_D",
        n_instances=20,
        eff_add_success_rate={p: 0.95 for p in good_preds}
        | {p: 0.55 for p in fragile_preds},
        eff_del_success_rate={"r_del": 0.90},
        overall_pass_rate=0.75,
        failure_signatures={"miss_add:r_fragile_1|miss_add:r_fragile_2": 6},
    )
    bank.add_or_update(contract_d, report_d)

    # ── Skill E: confused with D (stage2 diagnostics will flag) ──
    e_preds = {"r1", "r2", "r3", "r_unique_e"}

    for i in range(15):
        segments.append(_make_segment(
            _sid(), f"traj_{i % 4}", "skill_E",
            t_start=5000 + i * 70, t_end=5000 + i * 70 + 35,
            eff_add=e_preds, eff_del=set(), eff_event={"hover"},
        ))

    contract_e = SkillEffectsContract(
        skill_id="skill_E",
        eff_add=e_preds,
        eff_event={"hover"},
        support={p: 15 for p in e_preds} | {"hover": 15},
        n_instances=15,
    )
    report_e = VerificationReport(
        skill_id="skill_E",
        n_instances=15,
        eff_add_success_rate={p: 0.93 for p in e_preds},
        eff_event_rate={"hover": 1.0},
        overall_pass_rate=0.87,
    )
    bank.add_or_update(contract_e, report_e)

    stage2_diags = []
    for i in range(10):
        stage2_diags.append({
            "assigned_skill": "skill_E",
            "candidates": [
                {"skill": "skill_E", "total_score": 5.0},
                {"skill": "skill_D", "total_score": 4.5},
            ],
        })

    return bank, segments, stage2_diags


def main():
    bank, segments, stage2_diags = build_toy_data()

    config = BankMaintenanceConfig(
        split_pass_rate_thresh=0.70,
        merge_eff_jaccard_thresh=0.80,
        merge_emb_cosine_thresh=0.0,
        merge_transition_overlap_min=0.0,
        min_child_size=5,
        child_pass_rate_thresh=0.70,
        refine_drop_success_rate=0.60,
        lsh_threshold=0.40,
    )

    print(f"Bank before maintenance: {bank.skill_ids}")
    print(f"Segments: {len(segments)}")
    print()

    result = run_bank_maintenance(
        bank=bank,
        all_segments=segments,
        config=config,
        stage2_diagnostics=stage2_diags,
    )

    print("=" * 60)
    print("BANK MAINTENANCE DIFF REPORT")
    print("=" * 60)
    print(result.diff_report.summary())
    print()

    print("Bank after maintenance:", bank.skill_ids)
    print()

    print("Full result dict:")
    pprint(result.to_dict())


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    main()
